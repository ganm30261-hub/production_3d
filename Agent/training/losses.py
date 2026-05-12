# training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import PseudoLabelConfig


class DiceLoss(nn.Module):
    """
    二分类 Dice Loss，作用于 wall 类别（class=1）。

    Dice = 2 * |P ∩ T| / (|P| + |T|)
    Loss = 1 - Dice

    用 softmax 概率而非 sigmoid，和 CrossEntropyLoss 保持一致的
    输入格式 (B, C, H, W)，不需要额外转换。
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits : (B, num_classes, H, W)
        target : (B, H, W)  long，值为 0 或 1
        """
        prob  = torch.softmax(logits, dim=1)[:, 1].reshape(-1)   # wall 概率，展平
        tgt   = (target == 1).float().reshape(-1)
        inter = (prob * tgt).sum()
        return 1 - (2 * inter + self.smooth) / (prob.sum() + tgt.sum() + self.smooth)


class MultiTaskLoss(nn.Module):
    """
    分割损失 + 检测损失的加权组合。

    分割损失 = bce_weight * CrossEntropy + dice_weight * Dice
    检测损失 = RPN loss + RoI loss（由 Faster R-CNN 内部计算，直接求和）
    总损失   = 分割损失 + 检测损失

    wall_class_weight 对 wall 类别上调权重，缓解背景/墙体面积不均衡。
    """

    def __init__(self, cfg: PseudoLabelConfig):
        super().__init__()
        w = torch.tensor([1.0, cfg.wall_class_weight])
        self.ce   = nn.CrossEntropyLoss(weight=w, ignore_index=255)
        self.dice = DiceLoss()
        self.cfg  = cfg

    def forward(
        self,
        seg_logits:  torch.Tensor,
        seg_targets: torch.Tensor,
        det_losses:  dict,
    ) -> dict:
        """
        seg_logits  : (B, num_classes, H, W)
        seg_targets : (B, H, W)  long
        det_losses  : dict，来自 model 的 det_losses（训练模式下 Faster R-CNN 返回）
                      推理模式下传 {} 或 None

        返回 dict，包含各项损失值，方便 MLflow 分别记录：
        {
            'loss_total': Tensor  ← 唯一需要 .backward() 的
            'loss_seg'  : float
            'loss_bce'  : float
            'loss_dice' : float
            'loss_det'  : float
        }
        """
        l_bce  = self.ce(seg_logits, seg_targets)
        l_dice = self.dice(seg_logits, seg_targets)
        l_seg  = self.cfg.bce_weight * l_bce + self.cfg.dice_weight * l_dice

        l_det = (
            sum(det_losses.values())
            if det_losses
            else torch.tensor(0.0, device=seg_logits.device)
        )

        return {
            'loss_total': l_seg + l_det,
            'loss_seg':   l_seg.item(),
            'loss_bce':   l_bce.item(),
            'loss_dice':  l_dice.item(),
            'loss_det':   l_det.item() if det_losses else 0.0,
        }
