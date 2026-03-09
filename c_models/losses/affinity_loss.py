"""
Affinity Field Loss（生产级）

论文: "Adaptive Affinity Fields for Semantic Segmentation" (Ke et al., ECCV 2018)

通过学习相邻像素之间的亲和关系来改善分割边界清晰度。
传统损失（BCE/CE）独立处理每个像素，而 Affinity Loss 考虑邻域一致性：
同类像素亲和度高，异类像素亲和度低。

使用方法：
    loss_fn = AffinityFieldLoss(kernel_size=3)
    loss = loss_fn(pred_mask, target_mask)
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AffinityFieldLoss(nn.Module):
    """
    基础 Affinity Field Loss。

    计算预测与真实掩码在邻域内的亲和度差异。

    Args:
        kernel_size: 邻域大小（3 = 8 邻域, 5 = 24 邻域），必须为奇数
        reduction: 归约方式 ('mean', 'sum', 'none')
        boundary_weight: 边界像素额外权重（>1.0 时生效）
        use_log: True 用 BCE 计算亲和度差异，False 用 L1
    """

    def __init__(
        self,
        kernel_size: int = 3,
        reduction: str = 'mean',
        boundary_weight: float = 1.0,
        use_log: bool = False,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size 必须为奇数，收到 {kernel_size}")
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction 必须为 mean/sum/none，收到 {reduction}")

        self.kernel_size = kernel_size
        self.reduction = reduction
        self.boundary_weight = boundary_weight
        self.use_log = use_log
        self.offsets = self._generate_offsets()

    def _generate_offsets(self) -> List[Tuple[int, int]]:
        half = self.kernel_size // 2
        return [
            (dy, dx)
            for dy in range(-half, half + 1)
            for dx in range(-half, half + 1)
            if not (dy == 0 and dx == 0)
        ]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测掩码 (B, C, H, W)，值域 [0, 1]
            target: 真实掩码 (B, C, H, W)，值域 {0, 1}
            weight_map: 可选像素级权重 (B, 1, H, W)

        Returns:
            标量损失（reduction='mean'/'sum'）或逐像素损失
        """
        pred = self._ensure_4d(pred)
        target = self._ensure_4d(target)

        pred_aff = self._compute_affinity(pred)
        target_aff = self._compute_affinity(target)

        if self.use_log:
            loss = F.binary_cross_entropy(pred_aff, target_aff, reduction='none')
        else:
            loss = torch.abs(pred_aff - target_aff)

        if weight_map is not None:
            loss = loss * weight_map.expand_as(loss[:, :1]).repeat(1, loss.size(1), 1, 1)

        if self.boundary_weight > 1.0:
            boundary = self._detect_boundary(target)
            bw = torch.where(
                boundary,
                torch.tensor(self.boundary_weight, device=loss.device),
                torch.tensor(1.0, device=loss.device),
            )
            loss = loss * bw.unsqueeze(1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def _compute_affinity(self, x: torch.Tensor) -> torch.Tensor:
        """affinity(i,j) = 1 - |x[i] - x[j]|，沿所有偏移方向拼接。"""
        return torch.cat(
            [1.0 - torch.abs(x - self._shift(x, dy, dx)) for dy, dx in self.offsets],
            dim=1,
        )

    @staticmethod
    def _shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        B, C, H, W = x.shape
        padded = F.pad(
            x,
            (max(0, -dx), max(0, dx), max(0, -dy), max(0, dy)),
            mode='replicate',
        )
        ys = max(0, -dy) + dy
        xs = max(0, -dx) + dx
        return padded[:, :, ys:ys + H, xs:xs + W]

    @staticmethod
    def _detect_boundary(mask: torch.Tensor) -> torch.Tensor:
        """形态学梯度检测边界。"""
        dilated = F.max_pool2d(mask, 3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, 3, stride=1, padding=1)
        return ((dilated - eroded) > 0.5).squeeze(1)

    @staticmethod
    def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1) if x.dim() == 3 else x


class MultiScaleAffinityLoss(nn.Module):
    """
    多尺度 Affinity Loss。

    在不同下采样率上计算亲和度损失，捕获粗粒度和细粒度的边界信息。

    Args:
        scales: 下采样倍率列表（如 [1, 2, 4]）
        kernel_size: 基础邻域大小
        scale_weights: 各尺度权重（None 时自动按 1/scale 分配）
    """

    def __init__(
        self,
        scales: Optional[List[int]] = None,
        kernel_size: int = 3,
        scale_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.scales = scales or [1, 2, 4]

        if scale_weights is None:
            raw = [1.0 / s for s in self.scales]
        else:
            if len(scale_weights) != len(self.scales):
                raise ValueError("scale_weights 长度必须等于 scales 长度")
            raw = list(scale_weights)
        total = sum(raw)
        self.scale_weights = [w / total for w in raw]

        self.losses = nn.ModuleList(
            [AffinityFieldLoss(kernel_size=kernel_size) for _ in self.scales]
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total: torch.Tensor = torch.tensor(0.0, device=pred.device)
        for scale, weight, loss_fn in zip(self.scales, self.scale_weights, self.losses):
            if scale == 1:
                total = total + weight * loss_fn(pred, target)
            else:
                total = total + weight * loss_fn(
                    F.avg_pool2d(pred, scale),
                    F.avg_pool2d(target.float(), scale),
                )
        return total


class DirectionalAffinityLoss(nn.Module):
    """
    方向性 Affinity Loss。

    专为建筑平面图设计——墙体通常水平/垂直，因此对这两个方向加权更高。

    Args:
        horizontal_weight: 水平方向权重
        vertical_weight: 垂直方向权重
        diagonal_weight: 对角线方向权重
    """

    def __init__(
        self,
        horizontal_weight: float = 1.0,
        vertical_weight: float = 1.0,
        diagonal_weight: float = 0.5,
    ):
        super().__init__()
        self.h_w = horizontal_weight
        self.v_w = vertical_weight
        self.d_w = diagonal_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.unsqueeze(1) if pred.dim() == 3 else pred
        target = target.unsqueeze(1) if target.dim() == 3 else target

        total = torch.tensor(0.0, device=pred.device)
        weight_sum = 0.0

        if self.h_w > 0:
            total = total + self.h_w * self._dir_loss(pred, target, 0, 1)
            weight_sum += self.h_w
        if self.v_w > 0:
            total = total + self.v_w * self._dir_loss(pred, target, 1, 0)
            weight_sum += self.v_w
        if self.d_w > 0:
            d1 = self._dir_loss(pred, target, 1, 1)
            d2 = self._dir_loss(pred, target, 1, -1)
            total = total + self.d_w * (d1 + d2) / 2
            weight_sum += self.d_w

        return total / max(weight_sum, 1e-8)

    @staticmethod
    def _dir_loss(pred, target, dy, dx):
        ps = AffinityFieldLoss._shift(pred, dy, dx)
        ts = AffinityFieldLoss._shift(target, dy, dx)
        pa = 1.0 - torch.abs(pred - ps)
        ta = 1.0 - torch.abs(target.float() - ts.float())
        return F.binary_cross_entropy(pa, ta)
