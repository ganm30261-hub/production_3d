"""
门窗目标检测训练器 (论文 Section 2.2)

模型: Faster R-CNN + ResNet-50-FPN backbone (torchvision)
数据: data.pipeline.export_symbol_detection_data() 导出的 images/ + annotations.json
检测目标: window (label=1), door (label=2)

论文原文:
    "For the symbol detection task, we opted for a bounding-box detection model.
    For the architecture, we have chosen the popular Faster-RCNN model with a
    ResNet backbone."

使用方法:
    from models.symbol_det_trainer import SymbolDetectionTrainer, SymbolDetTrainerConfig

    trainer = SymbolDetectionTrainer(SymbolDetTrainerConfig(
        data_dir='output/symbol_detection',
    ))
    results = trainer.train()
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from .exceptions import TrainerError

logger = logging.getLogger(__name__)

# 可选依赖
try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    _HAS_DET = True
except ImportError:
    _HAS_DET = False


# ============================================================================
# Dataset
# ============================================================================

class SymbolDetectionDataset(Dataset):
    """
    加载 pipeline.export_symbol_detection_data() 导出的数据。

    目录结构:
        {data_dir}/{split}/images/000001.png
        {data_dir}/{split}/annotations.json

    annotations.json 格式:
        [{"image_id": 0, "file_name": "000001.png",
          "width": 800, "height": 600,
          "boxes": [[x1,y1,x2,y2], ...],
          "labels": [2, 1, ...]}, ...]

    Faster R-CNN 约定: 0=背景(内部), 1=window, 2=door
    """

    def __init__(self, data_dir: str, split: str = 'train'):
        self.images_dir = Path(data_dir) / split / 'images'
        ann_file = Path(data_dir) / split / 'annotations.json'

        if not ann_file.exists():
            raise TrainerError(f"标注文件不存在: {ann_file}")

        with open(ann_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        if not self.annotations:
            raise TrainerError(f"标注文件为空: {ann_file}")

        logger.info("SymbolDetDataset [%s]: %d 样本", split, len(self.annotations))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        # 图像 → (3, H, W) float [0, 1]
        image = cv2.imread(str(self.images_dir / ann['file_name']))
        if image is None:
            raise TrainerError(f"无法读取: {self.images_dir / ann['file_name']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # bbox + label
        boxes = ann.get('boxes', [])
        labels = ann.get('labels', [])

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        return image, {'boxes': boxes_t, 'labels': labels_t}


def _collate_fn(batch):
    """Faster R-CNN 需要 list[Tensor], list[dict] 格式。"""
    return [b[0] for b in batch], [b[1] for b in batch]


# ============================================================================
# 配置
# ============================================================================

@dataclass
class SymbolDetTrainerConfig:
    """
    门窗检测训练配置

    论文 Section 2.2:
      架构: Faster R-CNN + ResNet-50
      类别: 0=背景(内部), 1=window, 2=door → num_classes=3
    """
    data_dir: str = './output/symbol_detection'
    pretrained: bool = True
    num_classes: int = 3  # background + window + door

    # 训练
    num_epochs: int = 15
    batch_size: int = 2
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_step_size: int = 5
    lr_gamma: float = 0.1
    num_workers: int = 4
    device: str = 'auto'

    # 输出
    save_dir: str = 'runs/symbol_detection'
    save_best: bool = True


# ============================================================================
# 训练器
# ============================================================================

class SymbolDetectionTrainer:
    """
    门窗目标检测训练器。

    完整训练流程:
      1. 加载 images/ + annotations.json
      2. 创建 Faster R-CNN (替换分类头为 3 类)
      3. SGD 优化器 + StepLR 调度器
      4. 每个 epoch 验证 loss，保存最佳模型
    """

    def __init__(self, config: Optional[SymbolDetTrainerConfig] = None):
        if not _HAS_DET:
            raise TrainerError("torchvision detection 不可用: pip install torchvision")
        self.config = config or SymbolDetTrainerConfig()
        self.device = self._resolve_device()

    def _resolve_device(self):
        if self.config.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.config.device)

    def _create_model(self):
        """Faster R-CNN + ResNet-50-FPN, 替换分类头。"""
        model = fasterrcnn_resnet50_fpn(pretrained=self.config.pretrained)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.config.num_classes,
        )
        return model

    def train(self) -> Dict[str, Any]:
        """执行完整训练，返回 {'best_val_loss': float, 'history': list}。"""
        cfg = self.config

        # ── 数据 ──
        train_loader = DataLoader(
            SymbolDetectionDataset(cfg.data_dir, 'train'),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=_collate_fn, pin_memory=True,
        )
        val_loader = DataLoader(
            SymbolDetectionDataset(cfg.data_dir, 'val'),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=_collate_fn, pin_memory=True,
        )

        # ── 模型 ──
        model = self._create_model().to(self.device)

        # ── 优化器 ──
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma,
        )

        # ── 循环 ──
        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        best_loss, history = float('inf'), []

        logger.info("门窗检测训练: epochs=%d, batch=%d, device=%s",
                     cfg.num_epochs, cfg.batch_size, self.device)

        for epoch in range(1, cfg.num_epochs + 1):
            model.train()
            epoch_loss, n = 0.0, 0

            for images, targets in train_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                loss = sum(l for l in loss_dict.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n, 1)
            val_loss = self._validate(model, val_loader)

            history.append({'epoch': epoch, 'train_loss': avg_loss,
                            'val_loss': val_loss, 'lr': scheduler.get_last_lr()[0]})

            logger.info("Epoch %d/%d | train=%.4f | val=%.4f",
                         epoch, cfg.num_epochs, avg_loss, val_loss)

            if cfg.save_best and val_loss < best_loss:
                best_loss = val_loss
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'val_loss': val_loss}, save_dir / 'best.pth')
                logger.info("  -> best (loss=%.4f)", val_loss)

        torch.save({'model_state_dict': model.state_dict()}, save_dir / 'last.pth')
        logger.info("训练完成, best val_loss=%.4f", best_loss)
        return {'best_val_loss': best_loss, 'history': history}

    @torch.no_grad()
    def _validate(self, model, loader) -> float:
        model.train()  # Faster R-CNN 需要 train 模式才返回 loss
        total, n = 0.0, 0
        for images, targets in loader:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total += sum(l.item() for l in loss_dict.values())
            n += 1
        return total / max(n, 1)
