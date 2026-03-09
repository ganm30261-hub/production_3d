"""
墙体语义分割训练器 (论文 Section 2.3)

模型: FPN + ResNet backbone (backbone/resnet_fpn.py)
损失: BCE + Affinity Field Loss + Kendall 自动加权
数据: data.pipeline.export_wall_segmentation_data() 导出的 images/ + masks/

使用方法:
    from models.wall_seg_trainer import WallSegmentationTrainer, WallSegTrainerConfig

    trainer = WallSegmentationTrainer(WallSegTrainerConfig(
        data_dir='output/wall_segmentation',
    ))
    results = trainer.train()
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .losses.affinity_loss import AffinityFieldLoss
from .losses.kendall_loss import KendallMultiTaskLoss
from .exceptions import TrainerError

logger = logging.getLogger(__name__)


# ============================================================================
# Dataset
# ============================================================================

class WallSegmentationDataset(Dataset):
    """
    加载 pipeline.export_wall_segmentation_data() 导出的数据。

    目录结构:
        {data_dir}/{split}/images/000001.png   — 平面图 RGB
        {data_dir}/{split}/masks/000001.png    — 墙体二值掩码 (0/255)
    """

    def __init__(self, data_dir: str, split: str = 'train', target_size: int = 512):
        self.images_dir = Path(data_dir) / split / 'images'
        self.masks_dir = Path(data_dir) / split / 'masks'
        self.target_size = target_size

        if not self.images_dir.exists():
            raise TrainerError(f"图像目录不存在: {self.images_dir}")
        if not self.masks_dir.exists():
            raise TrainerError(f"掩码目录不存在: {self.masks_dir}")

        self.filenames = sorted([
            f.name for f in self.images_dir.glob('*.png')
            if (self.masks_dir / f.name).exists()
        ])
        if not self.filenames:
            raise TrainerError(f"未找到配对的 image/mask: {data_dir}/{split}")

        logger.info("WallSegDataset [%s]: %d 样本", split, len(self.filenames))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # 图像 → (3, H, W) float [0, 1]
        image = cv2.imread(str(self.images_dir / fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.target_size, self.target_size))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # 掩码 → (1, H, W) float {0, 1}
        mask = cv2.imread(str(self.masks_dir / fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.target_size, self.target_size),
                          interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        return image, mask


# ============================================================================
# 配置
# ============================================================================

@dataclass
class WallSegTrainerConfig:
    """
    墙体分割训练配置

    论文 Section 2.3:
      架构: FPN + ResNet
      损失: BCE + Affinity Field Loss
      权重: Kendall 不确定性自动学习
    """
    data_dir: str = './output/wall_segmentation'
    backbone: str = 'resnet50'
    pretrained: bool = True
    target_size: int = 512

    # 训练
    num_epochs: int = 50
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = 'auto'

    # 损失
    use_affinity: bool = True
    use_kendall: bool = True
    affinity_kernel: int = 3
    boundary_weight: float = 2.0

    # 输出
    save_dir: str = 'runs/wall_segmentation'
    save_best: bool = True


# ============================================================================
# 训练器
# ============================================================================

class WallSegmentationTrainer:
    """
    墙体语义分割训练器。

    完整训练流程:
      1. 加载 images/ + masks/ 配对数据
      2. 创建 ResNetFPNSegmentation 模型 (num_classes=1, 二值分割)
      3. BCE + Affinity + Kendall 组合损失
      4. Adam 优化器 + CosineAnnealing 调度器
      5. 每个 epoch 验证 IoU，保存最佳模型
    """

    def __init__(self, config: Optional[WallSegTrainerConfig] = None):
        self.config = config or WallSegTrainerConfig()
        self.device = self._resolve_device()

    def _resolve_device(self):
        if self.config.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.config.device)

    def train(self) -> Dict[str, Any]:
        """执行完整训练，返回 {'best_val_iou': float, 'history': list}。"""
        cfg = self.config

        # ── 数据 ──
        train_loader = DataLoader(
            WallSegmentationDataset(cfg.data_dir, 'train', cfg.target_size),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            WallSegmentationDataset(cfg.data_dir, 'val', cfg.target_size),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
        )

        # ── 模型 ──
        from .backbone.resnet_fpn import (
            ResNetFPNSegmentation, ResNetFPNConfig, ResNetVariant,
        )
        variant_map = {v.value: v for v in ResNetVariant}
        model = ResNetFPNSegmentation(
            num_classes=1,
            backbone_config=ResNetFPNConfig(
                variant=variant_map[cfg.backbone],
                pretrained=cfg.pretrained,
            ),
        ).to(self.device)

        # 多 GPU DataParallel
        gpu_ids = getattr(self, '_multi_gpu_ids', None)
        if gpu_ids and len(gpu_ids) > 1 and torch.cuda.is_available():
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
            logger.info("DataParallel: GPUs %s", gpu_ids)

        # ── 损失 ──
        bce = nn.BCEWithLogitsLoss()
        affinity = (
            AffinityFieldLoss(kernel_size=cfg.affinity_kernel,
                              boundary_weight=cfg.boundary_weight)
            if cfg.use_affinity else None
        )
        kendall = (
            KendallMultiTaskLoss(num_tasks=2 if cfg.use_affinity else 1).to(self.device)
            if cfg.use_kendall else None
        )

        # ── 优化器 ──
        params = list(model.parameters())
        if kendall is not None:
            params += list(kendall.parameters())
        optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

        # ── 循环 ──
        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        best_iou, history = 0.0, []

        logger.info("墙体分割训练: epochs=%d, batch=%d, device=%s",
                     cfg.num_epochs, cfg.batch_size, self.device)

        for epoch in range(1, cfg.num_epochs + 1):
            model.train()
            if kendall:
                kendall.train()
            epoch_loss = 0.0

            for images, masks in train_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                logits = model(images)

                l_bce = bce(logits, masks)
                l_aff = affinity(torch.sigmoid(logits), masks) if affinity else None

                if kendall and l_aff is not None:
                    loss = kendall([l_bce, l_aff])
                elif l_aff is not None:
                    loss = l_bce + 0.5 * l_aff
                else:
                    loss = l_bce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            val_iou = self._validate(model, val_loader)

            info = {'epoch': epoch, 'train_loss': avg_loss, 'val_iou': val_iou,
                    'lr': scheduler.get_last_lr()[0]}
            if kendall:
                info['kendall_weights'] = kendall.get_weights()
            history.append(info)

            logger.info("Epoch %d/%d | loss=%.4f | IoU=%.4f",
                         epoch, cfg.num_epochs, avg_loss, val_iou)

            if cfg.save_best and val_iou > best_iou:
                best_iou = val_iou
                raw_model = model.module if hasattr(model, 'module') else model
                ckpt = {'epoch': epoch, 'model_state_dict': raw_model.state_dict(),
                        'val_iou': val_iou}
                if kendall:
                    ckpt['kendall_state_dict'] = kendall.state_dict()
                torch.save(ckpt, save_dir / 'best.pth')
                logger.info("  -> best (IoU=%.4f)", val_iou)

        raw_model = model.module if hasattr(model, 'module') else model
        torch.save({'model_state_dict': raw_model.state_dict()}, save_dir / 'last.pth')
        logger.info("训练完成, best IoU=%.4f", best_iou)
        return {'best_val_iou': best_iou, 'history': history}

    @torch.no_grad()
    def _validate(self, model, loader) -> float:
        model.eval()
        iou_sum, n = 0.0, 0
        for images, masks in loader:
            images, masks = images.to(self.device), masks.to(self.device)
            preds = (torch.sigmoid(model(images)) > 0.5).float()
            inter = (preds * masks).sum(dim=(1, 2, 3))
            union = ((preds + masks) > 0).float().sum(dim=(1, 2, 3))
            iou_sum += (inter / (union + 1e-8)).sum().item()
            n += images.size(0)
        return iou_sum / max(n, 1)
