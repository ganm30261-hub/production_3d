"""
models — 模型、损失函数、训练器

子包:
    losses              Affinity / Kendall / 融合损失
    backbone            ResNet-FPN 特征提取器
    yolo_trainer        YOLOv8 增强训练器 (原有方案)
    wall_seg_trainer    墙体分割训练器 (论文 Section 2.3)
    symbol_det_trainer  门窗检测训练器 (论文 Section 2.2)
    training_pipeline   训练流水线编排器
"""

from .losses import (
    AffinityFieldLoss, MultiScaleAffinityLoss, DirectionalAffinityLoss,
    KendallMultiTaskLoss, KendallMultiTaskLossWithNames,
    GradNormMultiTaskLoss, FixedWeightMultiTaskLoss,
    CombinedLoss, FocalLossWithAffinity, SegmentationLossWithBoundary,
)
from .backbone import (
    ResNetFPN, ResNetFPNConfig, ResNetVariant,
    ResNetFPNSegmentation, create_resnet_fpn, create_segmentation_model,
)
from .yolo_trainer import (
    EnhancedYOLOTrainer, YOLOTrainerConfig, TrainerConfig, train_enhanced_yolo,
)
from .wall_seg_trainer import (
    WallSegmentationTrainer, WallSegTrainerConfig,
)
from .symbol_det_trainer import (
    SymbolDetectionTrainer, SymbolDetTrainerConfig,
)
from .training_pipeline import (
    TrainingPipeline, TrainingPipelineConfig,
)
from .exceptions import ModelError, BackboneError, LossError, TrainerError

__all__ = [
    # Losses
    "AffinityFieldLoss", "MultiScaleAffinityLoss", "DirectionalAffinityLoss",
    "KendallMultiTaskLoss", "KendallMultiTaskLossWithNames",
    "GradNormMultiTaskLoss", "FixedWeightMultiTaskLoss",
    "CombinedLoss", "FocalLossWithAffinity", "SegmentationLossWithBoundary",
    # Backbone
    "ResNetFPN", "ResNetFPNConfig", "ResNetVariant",
    "ResNetFPNSegmentation", "create_resnet_fpn", "create_segmentation_model",
    # YOLO Trainer
    "EnhancedYOLOTrainer", "YOLOTrainerConfig", "TrainerConfig", "train_enhanced_yolo",
    # Wall Segmentation Trainer
    "WallSegmentationTrainer", "WallSegTrainerConfig",
    # Symbol Detection Trainer
    "SymbolDetectionTrainer", "SymbolDetTrainerConfig",
    # Training Pipeline
    "TrainingPipeline", "TrainingPipelineConfig",
    # Exceptions
    "ModelError", "BackboneError", "LossError", "TrainerError",
]
