# training/trainer.py
import dataclasses
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import CFG, CKPT_DIR, MLFLOW_DIR, DEVICE, logger, PseudoLabelConfig
from model.encoder import save_lora_weights
from model.model import DINOv2LoRAModel
from data.dataset import FloorplanDataset, collate_fn
from training.losses import MultiTaskLoss


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

class AverageMeter:
    """滑动平均计数器，用于 epoch 内累积 loss / metric。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.count = 0.0

    def update(self, value: float, n: int = 1):
        self.sum   += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def wall_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算 wall 类别（class=1）的 IoU。

    pred   : (B, H, W)  long，argmax 后的预测
    target : (B, H, W)  long，GT mask
    """
    p, t = pred.view(-1), target.view(-1)
    tp   = ((p == 1) & (t == 1)).sum()
    fp   = ((p == 1) & (t == 0)).sum()
    fn   = ((p == 0) & (t == 1)).sum()
    denom = tp + fp + fn
    return (tp / denom).item() if denom > 0 else 0.0


# ══════════════════════════════════════════════════════════════
# Checkpoint Manager
# ══════════════════════════════════════════════════════════════

class LoRACheckpointManager:
    """
    DINOv2+LoRA 专用检查点管理器。

    文件名格式：
        {dataset_version}_bs{batch}_lr{lr}_ep{epoch:03d}_iou{score:.4f}.pth
    例如：
        combined_bs4_lr5e-5_ep015_iou0.7234.pth

    三类文件：
        Top-K 轮转 checkpoint  每个 epoch 存，超出 save_top_k 自动淘汰最差
        best.pth               覆盖写，始终是最新最优（固定名，方便快速找）
        best_ep{n}_iou{x}.pth  最优快照，不被覆盖（方便回溯历史最优点）

    断点续训：
        load_for_resume() 恢复 model / optimizer / epoch
    """

    def __init__(
        self,
        cfg:        PseudoLabelConfig,
        save_top_k: int = 3,
        monitor:    str = 'val_iou',
    ):
        self.cfg        = cfg
        self.save_top_k = save_top_k
        self.monitor    = monitor
        self.scores     = []   # [(score, Path), ...]
        self.ckpt_dir   = Path(cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        lr_str   = f'{cfg.learning_rate:.0e}'.replace('-0', '-')
        self.tag = f'{cfg.dataset_version}_bs{cfg.batch_size}_lr{lr_str}'

    # ── 内部工具 ──

    def _make_path(self, epoch: int, score: float) -> Path:
        return self.ckpt_dir / f'{self.tag}_ep{epoch:03d}_iou{score:.4f}.pth'

    def _build_payload(
        self,
        model:        torch.nn.Module,
        optimizer:    torch.optim.Optimizer,
        epoch:        int,
        metrics:      dict,
    ) -> dict:
        return {
            'epoch':           epoch,
            'dataset_version': self.cfg.dataset_version,
            'batch_size':      self.cfg.batch_size,
            'learning_rate':   self.cfg.learning_rate,
            'max_epochs':      self.cfg.max_epochs,
            'lora_r':          self.cfg.lora_r,
            'lora_alpha':      self.cfg.lora_alpha,
            'metrics':         metrics,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }

    # ── 公开 API ──

    def save(
        self,
        model:     torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch:     int,
        metrics:   dict,
    ) -> None:
        """保存当前 epoch checkpoint，维护 Top-K。"""
        score = metrics.get(self.monitor, 0.0)
        path  = self._make_path(epoch, score)
        torch.save(self._build_payload(model, optimizer, epoch, metrics), path)

        self.scores.append((score, path))
        self.scores.sort(key=lambda x: x[0], reverse=True)

        # 淘汰 Top-K 外最差的
        while len(self.scores) > self.save_top_k:
            _, old_path = self.scores.pop()
            if old_path.exists():
                old_path.unlink()
                logger.info(f'  [ckpt] 淘汰: {old_path.name}')

        # 检查当前 checkpoint 是否还在 Top-K 内
        in_topk = any(p == path for _, p in self.scores)
        if in_topk:
            rank = next(i + 1 for i, (_, p) in enumerate(self.scores) if p == path)
            logger.info(f'  [ckpt] 保存: {path.name}  (Top-{self.save_top_k} 第{rank}位)')
        else:
            logger.info(f'  [ckpt] 已淘汰（不在 Top-{self.save_top_k}）: {path.name}')

    def save_best(
        self,
        model:     torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch:     int,
        metrics:   dict,
    ) -> Path:
        """
        保存最优模型，同时输出三个文件：
            best.pth              固定名，覆盖写
            best_ep{n}_iou{x}.pth 快照，不被覆盖
            best_lora_only.pth    LoRA 增量（几 MB）
        """
        score     = metrics.get(self.monitor, 0.0)
        payload   = self._build_payload(model, optimizer, epoch, metrics)

        best_path = self.ckpt_dir / f'{self.tag}_best.pth'
        snap_path = self.ckpt_dir / f'{self.tag}_best_ep{epoch:03d}_iou{score:.4f}.pth'
        lora_path = self.ckpt_dir / f'{self.tag}_best_lora_only.pth'

        torch.save(payload, best_path)
        torch.save(payload, snap_path)
        save_lora_weights(model.encoder, str(lora_path))

        logger.info(f'  [best] {best_path.name}  iou={score:.4f}  ep{epoch}')
        logger.info(f'  [snap] {snap_path.name}')
        logger.info(f'  [lora] {lora_path.name}')
        return best_path

    def load_for_resume(
        self,
        resume_path: str,
        model:       torch.nn.Module,
        optimizer:   torch.optim.Optimizer,
    ) -> int:
        """
        从 checkpoint 恢复 model + optimizer。
        返回 start_epoch（下一个 epoch 编号）。
        """
        ckpt = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])

        start_epoch = ckpt['epoch'] + 1
        saved_iou   = ckpt.get('metrics', {}).get(self.monitor, 0.0)

        logger.info(f'  [resume] 从 {Path(resume_path).name} 恢复')
        logger.info(f'  [resume] 已完成 epoch={ckpt["epoch"]}  val_iou={saved_iou:.4f}')
        logger.info(f'  [resume] 继续从 epoch={start_epoch} 训练')

        # 重建 scores 列表（扫描目录内已有 Top-K 文件）
        for p in sorted(self.ckpt_dir.glob(f'{self.tag}_ep*.pth')):
            try:
                c = torch.load(p, map_location='cpu')
                s = c.get('metrics', {}).get(self.monitor, 0.0)
                self.scores.append((s, p))
            except Exception:
                pass
        self.scores.sort(key=lambda x: x[0], reverse=True)
        self.scores = self.scores[:self.save_top_k]
        logger.info(f'  [resume] 扫描到 {len(self.scores)} 个已有 checkpoint')

        return start_epoch


# ══════════════════════════════════════════════════════════════
# 训练主循环
# ══════════════════════════════════════════════════════════════

def train_one_version(
    version:  str,
    base_cfg: Optional[PseudoLabelConfig] = None,
) -> Optional[dict]:
    """
    对指定 dataset_version 完成一次完整训练。

    方式 A（单版本）:  train_one_version('hq')
    方式 B（全版本）:  for v in ('hq', 'hq_arch', 'combined'): train_one_version(v)
    断点续训          : base_cfg.resume_from = '/path/to/xxx.pth'

    返回:
    {
        'version', 'best_val_iou', 'best_ckpt',
        'ckpt_dir', 'gcs_dir', 'run_tag'
    }
    或 None（训练集为空时）
    """
    cfg = dataclasses.replace(base_cfg or CFG, dataset_version=version)

    logger.info('=' * 65)
    logger.info(f'开始训练  version={version}')
    logger.info(f'  train    : {cfg.train_files}')
    logger.info(f'  val      : {cfg.val_files}')
    logger.info(f'  ckpt_dir : {cfg.checkpoint_dir}')
    logger.info(f'  run_name : {cfg.mlflow_run_name}')
    logger.info(f'  resume   : {cfg.resume_from}')
    logger.info('=' * 65)

    # ── 数据 ──
    train_ds = FloorplanDataset(cfg, 'train')
    val_ds   = FloorplanDataset(cfg, 'val')

    if len(train_ds) == 0:
        logger.error(f'version={version} 训练集为空，检查 split 文件是否已生成')
        return None

    train_loader = DataLoader(
        train_ds, cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True, collate_fn=collate_fn,
    )
    logger.info(f'train batches={len(train_loader)}  val batches={len(val_loader)}')

    # ── 模型 ──
    model     = DINOv2LoRAModel(cfg).to(DEVICE)
    criterion = MultiTaskLoss(cfg).to(DEVICE)

    # 分层学习率：LoRA 参数用基础 lr，检测/分割头用 5x lr
    lora_params  = [p for n, p in model.named_parameters()
                    if p.requires_grad and 'lora_' in n]
    head_params  = [p for n, p in model.named_parameters()
                    if p.requires_grad and any(
                        k in n for k in ('seg_head', 'det_fpn', 'rpn', 'roi_heads')
                    )]
    other_params = [p for n, p in model.named_parameters()
                    if p.requires_grad
                    and p not in set(lora_params + head_params)]

    optimizer = torch.optim.AdamW([
        {'params': lora_params,  'lr': cfg.learning_rate,       'initial_lr': cfg.learning_rate,       'name': 'lora'},
        {'params': head_params,  'lr': cfg.learning_rate * 5.0, 'initial_lr': cfg.learning_rate * 5.0, 'name': 'head'},
        {'params': other_params, 'lr': cfg.learning_rate,       'initial_lr': cfg.learning_rate,       'name': 'other'},
    ], weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = max(cfg.max_epochs - cfg.warmup_epochs, 1),
        eta_min = cfg.learning_rate * 0.01,
    )
    scaler = torch.amp.GradScaler(
        'cuda', enabled=(cfg.use_amp and DEVICE == 'cuda')
    )

    # ── Checkpoint Manager ──
    ckpt_mgr    = LoRACheckpointManager(cfg, save_top_k=3)
    start_epoch = 1
    best_iou    = 0.0

    if cfg.resume_from and os.path.exists(cfg.resume_from):
        start_epoch = ckpt_mgr.load_for_resume(cfg.resume_from, model, optimizer)
    elif cfg.resume_from:
        logger.warning(f'resume_from 路径不存在，从头训练: {cfg.resume_from}')

    # ── MLflow ──
    mlflow.set_tracking_uri(f'file://{MLFLOW_DIR}')
    mlflow.set_experiment(cfg.mlflow_experiment)

    lr_str  = f'{cfg.learning_rate:.0e}'.replace('-0', '-')
    run_tag = (
        f'{version}'
        f'_ep{cfg.max_epochs}'
        f'_bs{cfg.batch_size}'
        f'_lr{lr_str}'
    )
    best_path: Optional[Path] = None

    with mlflow.start_run(run_name=cfg.mlflow_run_name) as run:
        mlflow.log_params({
            'dataset_version': version,
            'backbone':        cfg.dinov2_model,
            'lora_r':          cfg.lora_r,
            'lora_alpha':      cfg.lora_alpha,
            'lr':              cfg.learning_rate,
            'max_epochs':      cfg.max_epochs,
            'batch_size':      cfg.batch_size,
            'warmup_epochs':   cfg.warmup_epochs,
            'train_n':         len(train_ds),
            'val_n':           len(val_ds),
            'resume_from':     str(cfg.resume_from),
            'start_epoch':     start_epoch,
        })
        logger.info(f'MLflow run_id={run.info.run_id}')

        for epoch in range(start_epoch, cfg.max_epochs + 1):

            # ── Warmup：线性增大 lr ──
            if epoch <= cfg.warmup_epochs:
                scale = epoch / cfg.warmup_epochs
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['initial_lr'] * scale

            # ── Train ──
            model.train()
            criterion.train()
            meters = {k: AverageMeter() for k in ('total', 'seg', 'det')}
            t0 = time.time()

            for batch in tqdm(
                train_loader,
                desc=f'[{version}] Ep{epoch:03d} train',
                leave=False,
            ):
                images  = batch['image'].to(DEVICE)
                masks   = batch['mask'].to(DEVICE)
                targets = [
                    {'boxes': b.to(DEVICE), 'labels': l.to(DEVICE)}
                    for b, l in zip(batch['boxes'], batch['labels'])
                ]

                optimizer.zero_grad()
                with torch.amp.autocast(
                    device_type=DEVICE,
                    enabled=(cfg.use_amp and DEVICE == 'cuda'),
                ):
                    out = model(images, targets)
                    ld  = criterion(out['seg_logits'], masks, out['det_losses'])

                scaler.scale(ld['loss_total']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()

                n = images.size(0)
                meters['total'].update(ld['loss_total'].item(), n)
                meters['seg'].update(ld['loss_seg'], n)
                meters['det'].update(ld['loss_det'], n)

            # ── Val ──
            model.eval()
            iou_m = AverageMeter()

            with torch.no_grad():
                for batch in tqdm(
                    val_loader,
                    desc=f'[{version}] Ep{epoch:03d} val  ',
                    leave=False,
                ):
                    images = batch['image'].to(DEVICE)
                    masks  = batch['mask'].to(DEVICE)
                    out    = model(images)
                    preds  = out['seg_logits'].argmax(dim=1)
                    iou_m.update(wall_iou(preds.cpu(), masks.cpu()), images.size(0))

            if epoch > cfg.warmup_epochs:
                scheduler.step()

            lr = optimizer.param_groups[0]['lr']
            metrics = {
                'val_iou':    iou_m.avg,
                'train_loss': meters['total'].avg,
                'train_seg':  meters['seg'].avg,
                'train_det':  meters['det'].avg,
            }

            mlflow.log_metrics({**metrics, 'lr': lr}, step=epoch)

            elapsed = time.time() - t0
            logger.info(
                f'[{version}] Ep{epoch:03d}  '
                f'loss={meters["total"].avg:.4f}  '
                f'seg={meters["seg"].avg:.4f}  '
                f'det={meters["det"].avg:.4f}  '
                f'val_iou={iou_m.avg:.4f}  '
                f'lr={lr:.2e}  {elapsed:.0f}s'
            )

            # ── 每 epoch 保存 Top-K checkpoint ──
            ckpt_mgr.save(model, optimizer, epoch, metrics)

            # ── 更新最优 ──
            if iou_m.avg > best_iou:
                best_iou  = iou_m.avg
                best_path = ckpt_mgr.save_best(model, optimizer, epoch, metrics)
                mlflow.log_metric('best_val_iou', best_iou, step=epoch)
                mlflow.log_param('best_ckpt_name', best_path.name)

        logger.info(f'[{version}] 训练完成  best_val_iou={best_iou:.4f}')

        # ── 训练结束：保存结果 JSON ──
        results = {
            'model':           'DINOv2_ViT-L_LoRA',
            'dataset_version': version,
            'train_files':     cfg.train_files,
            'best_val_iou':    best_iou,
            'epochs_trained':  cfg.max_epochs,
            'start_epoch':     start_epoch,
            'batch_size':      cfg.batch_size,
            'learning_rate':   cfg.learning_rate,
            'lora_r':          cfg.lora_r,
            'lora_alpha':      cfg.lora_alpha,
            'best_ckpt_name':  best_path.name if best_path else None,
            'training_date':   time.strftime('%Y-%m-%d %H:%M:%S'),
            'mlflow_run_id':   run.info.run_id,
            'gcs_dir':         f'gs://{cfg.gcs_bucket}/{cfg.gcs_model_dir}/',
        }
        results_path = os.path.join(cfg.checkpoint_dir, f'{run_tag}_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'结果 JSON: {results_path}')

        # ── MLflow artifact ──
        if best_path:
            mlflow.log_artifact(str(best_path), 'model')
            mlflow.log_artifact(results_path,   'results')
            lora_path = ckpt_mgr.ckpt_dir / f'{ckpt_mgr.tag}_best_lora_only.pth'
            if lora_path.exists():
                mlflow.log_artifact(str(lora_path), 'model')
            mlflow.log_params({
                'results_file': Path(results_path).name,
                'gcs_dir':      results['gcs_dir'],
            })

        # ── GCS 上传 ──
        gcs_dir = f'gs://{cfg.gcs_bucket}/{cfg.gcs_model_dir}/'
        GSUTIL  = '/root/google-cloud-sdk/bin/gsutil'

        upload_list = []
        if best_path and best_path.exists():
            upload_list.append((str(best_path), f'{gcs_dir}{best_path.name}'))
            lora_path = ckpt_mgr.ckpt_dir / f'{ckpt_mgr.tag}_best_lora_only.pth'
            if lora_path.exists():
                upload_list.append((str(lora_path), f'{gcs_dir}{lora_path.name}'))
        upload_list.append((results_path, f'{gcs_dir}{Path(results_path).name}'))

        logger.info(f'GCS 上传目标: {gcs_dir}')
        for src, dst in upload_list:
            ret = subprocess.run(
                f'{GSUTIL} cp {src} {dst}',
                shell=True, capture_output=True, text=True,
                env={**os.environ,
                     'PATH': f'/root/google-cloud-sdk/bin:{os.environ["PATH"]}'},
            )
            symbol = '✓' if ret.returncode == 0 else '✗'
            logger.info(f'  {symbol} {dst.split("/")[-1]}')
            if ret.returncode != 0 and ret.stderr:
                logger.warning(f'    {ret.stderr.strip()[:100]}')

        # ── 本地版本索引（追加写，不覆盖已有记录）──
        index_path = os.path.join(CKPT_DIR, 'version_index.json')
        existing   = {}
        if os.path.exists(index_path):
            with open(index_path) as f:
                existing = json.load(f)

        existing[run_tag] = {
            'dataset_version': version,
            'gcs_dir':         gcs_dir,
            'best_ckpt':       best_path.name if best_path else None,
            'best_val_iou':    round(best_iou, 4),
            'mlflow_run_id':   run.info.run_id,
            'training_date':   results['training_date'],
        }
        with open(index_path, 'w') as f:
            json.dump(existing, f, indent=2)
        logger.info(f'版本索引更新: {index_path}  (共 {len(existing)} 条记录)')

    return {
        'version':      version,
        'best_val_iou': best_iou,
        'best_ckpt':    str(best_path) if best_path else None,
        'ckpt_dir':     str(ckpt_mgr.ckpt_dir),
        'gcs_dir':      gcs_dir if best_path else None,
        'run_tag':      run_tag,
    }
