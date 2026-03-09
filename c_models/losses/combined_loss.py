"""
融合损失函数（生产级）

整合分割损失、Affinity Loss、Kendall 多任务加权为统一接口。

典型用法：
    loss_fn = CombinedLoss(use_affinity=True, use_kendall=True)
    result = loss_fn(seg_loss=base_loss, pred_mask=pred, target_mask=target)
    result['total_loss'].backward()
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .affinity_loss import AffinityFieldLoss, DirectionalAffinityLoss
from .kendall_loss import KendallMultiTaskLossWithNames

logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """
    融合损失：主损失 + Affinity + Kendall 加权。

    Args:
        use_affinity: 是否启用 Affinity Loss
        use_kendall: 是否启用 Kendall 自动加权
        affinity_type: 'standard' 或 'directional'
        affinity_kernel: Affinity 邻域大小
        affinity_weight: 固定权重（不用 Kendall 时生效）
        boundary_weight: 边界像素额外权重
    """

    def __init__(
        self,
        use_affinity: bool = True,
        use_kendall: bool = True,
        affinity_type: str = 'standard',
        affinity_kernel: int = 3,
        affinity_weight: float = 0.5,
        boundary_weight: float = 2.0,
    ):
        super().__init__()
        self.use_affinity = use_affinity
        self.use_kendall = use_kendall
        self.affinity_weight = affinity_weight

        if use_affinity:
            if affinity_type == 'directional':
                self.affinity_loss = DirectionalAffinityLoss()
            else:
                self.affinity_loss = AffinityFieldLoss(
                    kernel_size=affinity_kernel,
                    boundary_weight=boundary_weight,
                )

        if use_kendall:
            names = ['main']
            if use_affinity:
                names.append('affinity')
            self.kendall = KendallMultiTaskLossWithNames(task_names=names)

    def forward(
        self,
        seg_loss: torch.Tensor,
        pred_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            seg_loss: 主分割损失（标量 tensor）
            pred_mask: 预测掩码 (B, C, H, W)，值域 [0,1]
            target_mask: 真实掩码 (B, C, H, W)

        Returns:
            {'total_loss', 'seg_loss', 'affinity_loss'?, 'weights'?}
        """
        out: Dict[str, torch.Tensor] = {'seg_loss': seg_loss}
        tasks = {'main': seg_loss}

        if self.use_affinity and pred_mask is not None and target_mask is not None:
            pm = self._prepare(pred_mask)
            tm = self._prepare(target_mask)
            aff = self.affinity_loss(pm, tm)
            out['affinity_loss'] = aff
            tasks['affinity'] = aff

        if self.use_kendall and len(tasks) > 1:
            out['total_loss'] = self.kendall(tasks)
            out['weights'] = self.kendall.get_named_weights()
        else:
            total = seg_loss
            if 'affinity_loss' in out:
                total = total + self.affinity_weight * out['affinity_loss']
            out['total_loss'] = total

        return out

    def get_weights(self) -> Optional[Dict[str, float]]:
        if self.use_kendall and hasattr(self, 'kendall'):
            return self.kendall.get_named_weights()
        return None

    @staticmethod
    def _prepare(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return x.unsqueeze(1) if x.dim() == 3 else x


class FocalLossWithAffinity(nn.Module):
    """
    Focal Loss + Affinity Loss。

    Focal 处理类别不平衡，Affinity 改善边界。

    Args:
        alpha: 正样本权重（标量或每类列表）
        gamma: 调节因子
        affinity_weight: Affinity 权重
    """

    def __init__(
        self,
        alpha: Union[float, List[float]] = 0.25,
        gamma: float = 2.0,
        use_affinity: bool = True,
        affinity_weight: float = 0.3,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.use_affinity = use_affinity
        self.affinity_weight = affinity_weight

        if isinstance(alpha, (list, tuple)):
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = alpha

        if use_affinity:
            self.affinity_loss = AffinityFieldLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        focal = self._focal(pred, target)
        out['focal_loss'] = focal

        if self.use_affinity:
            pp = torch.sigmoid(pred)
            if target.dim() == 3:
                to = F.one_hot(target.long(), pred.size(1)).permute(0, 3, 1, 2).float()
            else:
                to = target.float()
            aff = self.affinity_loss(pp, to)
            out['affinity_loss'] = aff
            out['total_loss'] = focal + self.affinity_weight * aff
        else:
            out['total_loss'] = focal

        return out

    def _focal(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
        p = torch.sigmoid(pred)
        pt = p * target + (1 - p) * (1 - target)
        fw = (1 - pt) ** self.gamma

        if isinstance(self.alpha, torch.Tensor):
            at = self.alpha.view(1, -1, 1, 1) * target + (
                1 - self.alpha.view(1, -1, 1, 1)
            ) * (1 - target)
        else:
            at = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss = at * fw * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SegmentationLossWithBoundary(nn.Module):
    """
    分割损失 + 边界增强。

    专为平面图设计，强调墙体边界准确性。

    Args:
        base_loss: 'bce', 'ce', 'focal'
        boundary_loss: 'affinity', 'dice_boundary'
        base_weight / boundary_weight: 两部分权重
    """

    def __init__(
        self,
        base_loss: str = 'bce',
        boundary_loss: str = 'affinity',
        base_weight: float = 1.0,
        boundary_weight: float = 0.5,
        num_classes: int = 1,
    ):
        super().__init__()
        self.base_weight = base_weight
        self.boundary_weight = boundary_weight
        self.base_loss_type = base_loss

        if base_loss == 'focal':
            self._base_fn = self._focal
        elif base_loss == 'ce':
            self._base_fn = nn.CrossEntropyLoss()
        else:
            self._base_fn = nn.BCEWithLogitsLoss()

        self.boundary_loss_type = boundary_loss
        if boundary_loss == 'affinity':
            self._boundary_fn = DirectionalAffinityLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        if self.base_loss_type == 'ce' and target.dim() == pred.dim() - 1:
            bl = self._base_fn(pred, target.long())
        else:
            bl = self._base_fn(pred, target.float())
        out['base_loss'] = bl

        if self.boundary_loss_type == 'affinity':
            bdl = self._boundary_fn(torch.sigmoid(pred), target.float())
        else:
            bdl = self._dice_boundary(pred, target)
        out['boundary_loss'] = bdl
        out['total_loss'] = self.base_weight * bl + self.boundary_weight * bdl
        return out

    @staticmethod
    def _focal(pred, target, gamma=2.0):
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
        return ((1 - torch.exp(-bce)) ** gamma * bce).mean()

    @staticmethod
    def _dice_boundary(pred, target):
        pp = torch.sigmoid(pred)
        pb = SegmentationLossWithBoundary._boundary(pp)
        tb = SegmentationLossWithBoundary._boundary(target.float())
        inter = (pb * tb).sum()
        union = pb.sum() + tb.sum()
        return 1 - (2 * inter + 1e-6) / (union + 1e-6)

    @staticmethod
    def _boundary(mask, ks=3):
        p = ks // 2
        return F.max_pool2d(mask, ks, stride=1, padding=p) - (
            -F.max_pool2d(-mask, ks, stride=1, padding=p)
        )
