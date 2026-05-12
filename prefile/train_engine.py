"""
train_engine.py — 数据预处理 + 训练主引擎

职责：
  - FloorplanDataset：加载 / 增强 / 切 tile，返回 PyTorch 样本
  - 评估指标工具：AverageMeter / compute_wall_iou / compute_det_metrics
  - CheckpointManager：保存 Top-K checkpoint + best_model.pth
  - train_epoch / val_epoch：单 epoch 的训练和验证逻辑
  - main()：完整训练主循环（含 MLflow 记录 + GCS 上传）

入口：
  python train_engine.py
"""

import gc
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from numpy import genfromtxt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import PaperConfig, CUBICASA_ROOT, MLFLOW_DIR, CHECKPOINT_DIR
from model_arch import FloorplanModel, SegmentationLoss, build_model
from utils import (
    load_sample, adaptive_preprocess, load_preprocessed_from_gcs,
    crop_to_wall_bbox, augment_sample,
    extract_wall_mask, extract_door_window_boxes,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

class FloorplanDataset(Dataset):
    """
    联合数据集：同时提供墙体分割标注和门窗检测标注
    实现论文的完整预处理流程
    """

    def __init__(self, cfg: PaperConfig, split: str = 'train'):
        self.cfg      = cfg
        self.is_train = (split == 'train')
        data_file     = cfg.train_file if self.is_train else cfg.val_file
        full_path     = cfg.data_folder + data_file

        if full_path.startswith('gs://'):
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            with fs.open(full_path, 'r') as f:
                self.folders = np.array(f.read().splitlines())
        else:
            self.folders = genfromtxt(full_path, dtype='str')

        logger.info(f'{split} Dataset: {len(self.folders)} 个样本')

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx: int) -> dict:
        folder = self.folders[idx]
        try:
            sample = load_sample(self.cfg.data_folder, folder, CUBICASA_ROOT)
            seg    = sample['seg']   # (C, H, W)

            # 优先从 GCS 预处理缓存读取图片
            if self.cfg.use_preprocessed:
                image = load_preprocessed_from_gcs(
                    folder, self.cfg.gcs_bucket, self.cfg.gcs_preprocessed
                )
                if image is None:
                    image = sample['image']
                    if self.cfg.preprocess_enabled:
                        image = adaptive_preprocess(image, self.cfg.image_type)
            else:
                image = sample['image']
                if self.cfg.preprocess_enabled:
                    image = adaptive_preprocess(image, self.cfg.image_type)

            # 论文 Section 3：裁剪到 wall bbox
            if self.cfg.crop_to_wall:
                image, seg = crop_to_wall_bbox(
                    image, seg,
                    self.cfg.wall_class_id,
                    self.cfg.crop_padding,
                    self.cfg.min_crop_size,
                )
                if image is None:
                    return self._empty()

            # 数据增强（仅训练集，Section 2.1）
            if self.is_train:
                image, seg = augment_sample(
                    image, seg,
                    self.cfg.aug_scale_range,
                    self.cfg.aug_rotation_degrees,
                    self.cfg.aug_use_flip,
                )

            # 滑动窗口：随机取一个 tile（Section 2.1）
            h, w   = image.shape[:2]
            ts     = self.cfg.tile_size
            stride = ts - self.cfg.tile_overlap

            ys = list(range(0, max(h - ts + 1, 1), stride))
            xs = list(range(0, max(w - ts + 1, 1), stride))
            if not ys or ys[-1] + ts < h: ys.append(max(h - ts, 0))
            if not xs or xs[-1] + ts < w: xs.append(max(w - ts, 0))

            ty = random.choice(list(set(ys)))
            tx = random.choice(list(set(xs)))

            img_tile = image[ty:ty + ts, tx:tx + ts].copy()
            seg_tile = seg[:, ty:ty + ts, tx:tx + ts].copy()

            # Padding 补齐不足 tile_size 的情况
            th, tw = img_tile.shape[:2]
            if th < ts or tw < ts:
                img_tile = cv2.copyMakeBorder(
                    img_tile, 0, ts - th, 0, ts - tw, cv2.BORDER_REFLECT_101
                )
                new_seg  = np.zeros((seg_tile.shape[0], ts, ts), dtype=seg_tile.dtype)
                new_seg[:, :th, :tw] = seg_tile
                seg_tile = new_seg

            img_tile = cv2.resize(img_tile, (ts, ts), interpolation=cv2.INTER_LINEAR)
            seg_r    = np.zeros((seg_tile.shape[0], ts, ts), dtype=seg_tile.dtype)
            for c in range(seg_tile.shape[0]):
                seg_r[c] = cv2.resize(
                    seg_tile[c].astype(np.float32), (ts, ts),
                    interpolation=cv2.INTER_NEAREST
                ).astype(seg_tile.dtype)
            seg_tile = seg_r

            wall_mask        = extract_wall_mask(seg_tile, self.cfg.wall_class_id)
            boxes, labels    = extract_door_window_boxes(
                seg_tile,
                self.cfg.door_class_id,
                self.cfg.window_class_id,
                self.cfg.min_bbox_area,
            )

            img_t   = transforms.ToTensor()(img_tile)
            img_t   = transforms.Normalize(self.cfg.norm_mean, self.cfg.norm_std)(img_t)
            mask_t  = torch.from_numpy(wall_mask).long()
            boxes_t = torch.from_numpy(boxes)
            lbls_t  = torch.from_numpy(labels)

            del sample
            gc.collect()

            return {'image': img_t, 'mask': mask_t, 'boxes': boxes_t, 'labels': lbls_t}

        except Exception as e:
            logger.warning(f'样本 {folder} 加载失败: {e}')
            return self._empty()

    def _empty(self):
        ts = self.cfg.tile_size
        return {
            'image':  torch.zeros(3, ts, ts),
            'mask':   torch.zeros(ts, ts).long(),
            'boxes':  torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,),   dtype=torch.int64),
        }


def collate_fn(batch):
    """
    Faster R-CNN 要求每个样本的 boxes 数量可不同，
    故 boxes / labels 保持为 list，不 stack。
    """
    images  = torch.stack([b['image']  for b in batch])
    masks   = torch.stack([b['mask']   for b in batch])
    boxes   = [b['boxes']  for b in batch]
    labels  = [b['labels'] for b in batch]
    return {'image': images, 'mask': masks, 'boxes': boxes, 'labels': labels}


# ══════════════════════════════════════════════════════════════
# 评估指标
# ══════════════════════════════════════════════════════════════

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self):    self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def compute_wall_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """论文主要指标：墙体类别（class=1）的 IoU"""
    pred   = pred.view(-1)
    target = target.view(-1)
    tp    = ((pred == 1) & (target == 1)).sum().item()
    fp    = ((pred == 1) & (target == 0)).sum().item()
    fn    = ((pred == 0) & (target == 1)).sum().item()
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def compute_det_metrics(
    outputs:        List[dict],
    targets_boxes:  List[torch.Tensor],
    targets_labels: List[torch.Tensor],
    iou_threshold:  float = 0.5,
) -> dict:
    """计算检测任务的 precision / recall / F1"""
    from torchvision.ops import box_iou

    tp_total = fp_total = fn_total = 0

    for output, gt_boxes, gt_labels in zip(outputs, targets_boxes, targets_labels):
        if len(gt_boxes) == 0:
            fp_total += len(output.get('boxes', []))
            continue

        pred_boxes  = output.get('boxes',  torch.zeros((0, 4)))
        pred_scores = output.get('scores', torch.zeros(0))
        keep        = pred_scores > 0.5
        pred_boxes  = pred_boxes[keep]

        if len(pred_boxes) == 0:
            fn_total += len(gt_boxes)
            continue

        iou_mat    = box_iou(pred_boxes.cpu(), gt_boxes.cpu())
        matched_gt = set()
        for i in range(len(pred_boxes)):
            if iou_mat.shape[1] == 0:
                fp_total += 1
                continue
            max_iou, max_j = iou_mat[i].max(0)
            if max_iou.item() >= iou_threshold and max_j.item() not in matched_gt:
                tp_total += 1
                matched_gt.add(max_j.item())
            else:
                fp_total += 1
        fn_total += len(gt_boxes) - len(matched_gt)

    precision = tp_total / (tp_total + fp_total + 1e-8)
    recall    = tp_total / (tp_total + fn_total + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return {'det_precision': precision, 'det_recall': recall, 'det_f1': f1}


# ══════════════════════════════════════════════════════════════
# CheckpointManager
# ══════════════════════════════════════════════════════════════

class CheckpointManager:
    def __init__(self, ckpt_dir: str, save_top_k: int = 3,
                 monitor: str = 'val_wall_iou'):
        self.ckpt_dir   = Path(ckpt_dir)
        self.save_top_k = save_top_k
        self.monitor    = monitor
        self.scores     = []
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, seg_criterion, epoch, metrics):
        score = metrics.get(self.monitor, 0.0)
        path  = self.ckpt_dir / f'ep{epoch:03d}_iou{score:.4f}.pth'
        torch.save({
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'kendall_state':   seg_criterion.kendall.state_dict(),
            'metrics':         metrics,
        }, path)
        self.scores.append((score, path))
        self.scores.sort(key=lambda x: x[0], reverse=True)
        while len(self.scores) > self.save_top_k:
            _, old_path = self.scores.pop()
            if old_path.exists():
                old_path.unlink()
        logger.info(f'Checkpoint saved: {path.name}  (top-{self.save_top_k})')


# ══════════════════════════════════════════════════════════════
# 训练 / 验证 函数
# ══════════════════════════════════════════════════════════════

def train_epoch(
    model, seg_criterion, optimizer, scaler,
    loader, device, epoch, warmup_epochs, base_lr,
):
    model.train()
    seg_criterion.train()

    meters = {k: AverageMeter() for k in [
        'loss_total', 'loss_seg', 'loss_bce', 'loss_aff',
        'loss_rpn_cls', 'loss_rpn_box', 'loss_cls', 'loss_box',
    ]}
    t0 = time.time()

    # Warmup LR
    if epoch <= warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    for batch in loader:
        images  = batch['image'].to(device)
        masks   = batch['mask'].to(device)
        targets = [
            {'boxes': b.to(device), 'labels': l.to(device)}
            for b, l in zip(batch['boxes'], batch['labels'])
        ]

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device, enabled=scaler.is_enabled()):
            outputs    = model(images, targets)
            seg_losses = seg_criterion(outputs['seg_logits'], masks)
            loss_seg   = seg_losses['loss_seg']
            det_losses = outputs['det_losses']
            loss_det   = sum(det_losses.values())
            loss_total = loss_seg + loss_det

        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        n = images.size(0)
        meters['loss_total'].update(loss_total.item(), n)
        meters['loss_seg'].update(loss_seg.item(), n)
        meters['loss_bce'].update(seg_losses['loss_bce'], n)
        meters['loss_aff'].update(seg_losses['loss_aff'], n)
        for k in ['loss_rpn_cls', 'loss_rpn_box', 'loss_cls', 'loss_box']:
            if k in det_losses:
                meters[k].update(det_losses[k].item(), n)

    return {k: m.avg for k, m in meters.items()}, time.time() - t0


@torch.no_grad()
def val_epoch(model, seg_criterion, loader, device):
    model.eval()
    seg_criterion.eval()

    seg_loss_m = AverageMeter()
    wall_iou_m = AverageMeter()
    all_outputs, all_gt_boxes, all_gt_labels = [], [], []

    for batch in loader:
        images = batch['image'].to(device)
        masks  = batch['mask'].to(device)

        outputs    = model(images)
        seg_losses = seg_criterion(outputs['seg_logits'], masks)
        preds      = outputs['seg_logits'].argmax(dim=1)
        wall_iou   = compute_wall_iou(preds.cpu(), masks.cpu())

        n = images.size(0)
        seg_loss_m.update(seg_losses['loss_seg'].item(), n)
        wall_iou_m.update(wall_iou, n)

        all_outputs.extend(outputs['det_outputs'])
        all_gt_boxes.extend(batch['boxes'])
        all_gt_labels.extend(batch['labels'])

    det_metrics = compute_det_metrics(all_outputs, all_gt_boxes, all_gt_labels)

    return {
        'val_seg_loss': seg_loss_m.avg,
        'val_wall_iou': wall_iou_m.avg,
        **det_metrics,
    }


# ══════════════════════════════════════════════════════════════
# 主训练循环
# ══════════════════════════════════════════════════════════════

def main():
    import mlflow

    # ── 环境准备 ──
    os.chdir(CUBICASA_ROOT)
    sys.path.insert(0, CUBICASA_ROOT)

    CFG = PaperConfig()
    CFG.validate()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'device={DEVICE}  batch={CFG.batch_size}  epochs={CFG.max_epochs}')

    # ── 数据集 ──
    train_ds = FloorplanDataset(CFG, split='train')
    val_ds   = FloorplanDataset(CFG, split='val')

    train_loader = DataLoader(
        train_ds, batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=True,
        drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )
    logger.info(f'train batches={len(train_loader)}  val batches={len(val_loader)}')

    # ── 模型 ──
    model = build_model(CFG, DEVICE)

    # ── 损失 ──
    seg_criterion = SegmentationLoss(CFG).to(DEVICE)

    # ── 优化器 ──
    all_params = list(model.parameters()) + list(seg_criterion.kendall.parameters())
    optimizer  = torch.optim.AdamW(all_params, lr=CFG.learning_rate,
                                    weight_decay=CFG.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG.max_epochs - CFG.warmup_epochs,
        eta_min=CFG.learning_rate * 0.01,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=CFG.use_amp)

    # ── 检查点管理 ──
    ckpt_manager  = CheckpointManager(CFG.checkpoint_dir, CFG.save_top_k)
    best_wall_iou = 0.0

    # ── MLflow ──
    mlflow.set_tracking_uri(f'file://{MLFLOW_DIR}')
    mlflow.set_experiment(CFG.mlflow_experiment)

    with mlflow.start_run(run_name=CFG.mlflow_run_name) as run:
        mlflow.log_params({
            'model':                 'SharedResNet50_FPN_FasterRCNN',
            'pretrained':            CFG.pretrained_backbone,
            'fpn_channels':          CFG.fpn_out_channels,
            'batch_size':            CFG.batch_size,
            'max_epochs':            CFG.max_epochs,
            'learning_rate':         CFG.learning_rate,
            'weight_decay':          CFG.weight_decay,
            'warmup_epochs':         CFG.warmup_epochs,
            'tile_size':             CFG.tile_size,
            'crop_to_wall':          CFG.crop_to_wall,
            'wall_class_weight':     CFG.wall_class_weight,
            'affinity_neighborhood': CFG.affinity_neighborhood,
            'aug_scale_min':         CFG.aug_scale_range[0],
            'aug_scale_max':         CFG.aug_scale_range[1],
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        })
        logger.info(f'MLflow Run ID: {run.info.run_id}')

        header = (f'{"Epoch":>6}  {"total":>8}  {"seg":>8}  {"bce":>7}  {"aff":>7}  '
                  f'{"wall_iou":>9}  {"det_f1":>7}  {"lr":>8}  {"time":>6}')
        logger.info(header)

        for epoch in range(1, CFG.max_epochs + 1):
            train_metrics, elapsed = train_epoch(
                model, seg_criterion, optimizer, scaler,
                train_loader, DEVICE, epoch,
                CFG.warmup_epochs, CFG.learning_rate,
            )
            val_metrics = val_epoch(model, seg_criterion, val_loader, DEVICE)

            if epoch > CFG.warmup_epochs:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            mlflow.log_metrics({
                'train_loss_total':   train_metrics['loss_total'],
                'train_loss_seg':     train_metrics['loss_seg'],
                'train_loss_bce':     train_metrics['loss_bce'],
                'train_loss_aff':     train_metrics['loss_aff'],
                'train_loss_rpn_cls': train_metrics.get('loss_rpn_cls', 0),
                'train_loss_rpn_box': train_metrics.get('loss_rpn_box', 0),
                'train_loss_cls':     train_metrics.get('loss_cls', 0),
                'train_loss_box':     train_metrics.get('loss_box', 0),
                'val_seg_loss':       val_metrics['val_seg_loss'],
                'val_wall_iou':       val_metrics['val_wall_iou'],
                'det_precision':      val_metrics['det_precision'],
                'det_recall':         val_metrics['det_recall'],
                'det_f1':             val_metrics['det_f1'],
                'kendall_log_var_bce': seg_criterion.kendall.log_var_bce.item(),
                'kendall_log_var_aff': seg_criterion.kendall.log_var_aff.item(),
                'learning_rate':      current_lr,
            }, step=epoch)

            ckpt_manager.save(model, optimizer, seg_criterion, epoch,
                              {**train_metrics, **val_metrics})

            # 保存最优模型
            if val_metrics['val_wall_iou'] > best_wall_iou:
                best_wall_iou = val_metrics['val_wall_iou']
                best_path = f'{CFG.checkpoint_dir}/best_model.pth'
                torch.save({
                    'model_state':   model.state_dict(),
                    'kendall_state': seg_criterion.kendall.state_dict(),
                    'epoch':         epoch,
                    'val_wall_iou':  best_wall_iou,
                    'det_f1':        val_metrics['det_f1'],
                }, best_path)
                mlflow.log_metric('best_wall_iou', best_wall_iou, step=epoch)

            logger.info(
                f'{epoch:>6}  '
                f'{train_metrics["loss_total"]:>8.4f}  '
                f'{train_metrics["loss_seg"]:>8.4f}  '
                f'{train_metrics["loss_bce"]:>7.4f}  '
                f'{train_metrics["loss_aff"]:>7.4f}  '
                f'{val_metrics["val_wall_iou"]:>9.4f}  '
                f'{val_metrics["det_f1"]:>7.4f}  '
                f'{current_lr:>8.6f}  {elapsed:>5.1f}s'
            )

        # ── 保存结果 + 上传 GCS ──
        results = {
            'model':          'SharedResNet50_FPN_FasterRCNN',
            'best_wall_iou':  best_wall_iou,
            'paper_target':   0.81,
            'epochs_trained': CFG.max_epochs,
            'training_date':  time.strftime('%Y-%m-%d'),
            'mlflow_run_id':  run.info.run_id,
            'kendall_weights': {
                'log_var_bce': seg_criterion.kendall.log_var_bce.item(),
                'log_var_aff': seg_criterion.kendall.log_var_aff.item(),
                'w_bce': torch.exp(-seg_criterion.kendall.log_var_bce).item(),
                'w_aff': torch.exp(-seg_criterion.kendall.log_var_aff).item(),
            },
        }
        results_path = f'{CFG.checkpoint_dir}/training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        mlflow.log_artifact(f'{CFG.checkpoint_dir}/best_model.pth', 'model')
        mlflow.log_artifact(results_path, 'results')

        gcs_path = f'gs://{CFG.gcs_bucket}/{CFG.gcs_model_dir}/'
        for src, dst in [
            (f'{CFG.checkpoint_dir}/best_model.pth', f'{gcs_path}best_model.pth'),
            (results_path, f'{gcs_path}training_results.json'),
        ]:
            ret = subprocess.run(f'gsutil cp {src} {dst}',
                                 shell=True, capture_output=True, text=True)
            symbol = '✓' if ret.returncode == 0 else '✗'
            logger.info(f'{symbol} GCS upload: {dst.split("/")[-1]}')

        logger.info(f'训练完成！best_wall_iou={best_wall_iou:.4f}  (论文目标: 0.81)')


if __name__ == '__main__':
    main()
