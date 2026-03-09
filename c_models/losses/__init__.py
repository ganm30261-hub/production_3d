"""
losses — 损失函数子包
"""

from .affinity_loss import (
    AffinityFieldLoss,
    MultiScaleAffinityLoss,
    DirectionalAffinityLoss,
)
from .kendall_loss import (
    KendallMultiTaskLoss,
    KendallMultiTaskLossWithNames,
    GradNormMultiTaskLoss,
    FixedWeightMultiTaskLoss,
)
from .combined_loss import (
    CombinedLoss,
    FocalLossWithAffinity,
    SegmentationLossWithBoundary,
)

__all__ = [
    "AffinityFieldLoss", "MultiScaleAffinityLoss", "DirectionalAffinityLoss",
    "KendallMultiTaskLoss", "KendallMultiTaskLossWithNames",
    "GradNormMultiTaskLoss", "FixedWeightMultiTaskLoss",
    "CombinedLoss", "FocalLossWithAffinity", "SegmentationLossWithBoundary",
]
