"""
Kendall Multi-Task Loss Weighting（生产级）

论文: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
      (Kendall et al., CVPR 2018)

使用同方差不确定性自动学习多任务损失权重：
    L_total = Σ (1/(2σ²)) · L_i + log(σ_i)
不确定性高的任务权重自动降低，反之升高。

使用方法：
    mtl = KendallMultiTaskLoss(num_tasks=3)
    total = mtl([loss_seg, loss_det, loss_aff])
"""

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class KendallMultiTaskLoss(nn.Module):
    """
    Kendall 多任务损失加权。

    Args:
        num_tasks: 任务数量
        init_log_vars: 初始 log(σ²) 值列表（None 时全部初始化为 0）
        learn_weights: 是否通过反向传播学习权重
    """

    def __init__(
        self,
        num_tasks: int,
        init_log_vars: Optional[List[float]] = None,
        learn_weights: bool = True,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.learn_weights = learn_weights

        vals = init_log_vars or [0.0] * num_tasks
        if len(vals) != num_tasks:
            raise ValueError(
                f"init_log_vars 长度 ({len(vals)}) != num_tasks ({num_tasks})"
            )

        t = torch.tensor(vals, dtype=torch.float32)
        if learn_weights:
            self.log_vars = nn.Parameter(t)
        else:
            self.register_buffer('log_vars', t)

    def forward(
        self, losses: Union[List[torch.Tensor], Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        if isinstance(losses, dict):
            losses = list(losses.values())
        if len(losses) != self.num_tasks:
            raise ValueError(
                f"损失数量 ({len(losses)}) != num_tasks ({self.num_tasks})"
            )

        total = torch.tensor(0.0, device=self.log_vars.device)
        for i, loss_i in enumerate(losses):
            precision = 0.5 * torch.exp(-self.log_vars[i])
            reg = 0.5 * self.log_vars[i]
            total = total + precision * loss_i + reg
        return total

    def get_weights(self) -> List[float]:
        """当前精度权重 1/(2σ²)。"""
        with torch.no_grad():
            return [0.5 * torch.exp(-lv).item() for lv in self.log_vars]

    def get_uncertainties(self) -> List[float]:
        """当前不确定性 σ²。"""
        with torch.no_grad():
            return [torch.exp(lv).item() for lv in self.log_vars]

    def extra_repr(self) -> str:
        w = [f"{v:.3f}" for v in self.get_weights()]
        return f"num_tasks={self.num_tasks}, learn={self.learn_weights}, weights={w}"


class KendallMultiTaskLossWithNames(KendallMultiTaskLoss):
    """
    带任务名称的 Kendall 损失——便于日志记录和 TensorBoard/MLflow 集成。

    Args:
        task_names: 任务名称列表（决定任务数量）
    """

    def __init__(
        self,
        task_names: List[str],
        init_log_vars: Optional[List[float]] = None,
        learn_weights: bool = True,
    ):
        super().__init__(len(task_names), init_log_vars, learn_weights)
        self.task_names = list(task_names)

    def forward(
        self, losses: Union[List[torch.Tensor], Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        if isinstance(losses, dict):
            ordered = []
            for name in self.task_names:
                if name not in losses:
                    raise KeyError(f"缺少任务 '{name}' 的损失")
                ordered.append(losses[name])
            losses = ordered
        return super().forward(losses)

    def get_named_weights(self) -> Dict[str, float]:
        return dict(zip(self.task_names, self.get_weights()))

    def get_named_uncertainties(self) -> Dict[str, float]:
        return dict(zip(self.task_names, self.get_uncertainties()))

    def log_to_tensorboard(self, writer, step: int, prefix: str = 'mtl/'):
        for name, w in self.get_named_weights().items():
            writer.add_scalar(f'{prefix}weight/{name}', w, step)
        for name, u in self.get_named_uncertainties().items():
            writer.add_scalar(f'{prefix}uncertainty/{name}', u, step)


class GradNormMultiTaskLoss(nn.Module):
    """
    GradNorm 多任务加权。

    论文: "GradNorm: Gradient Normalization for Adaptive Loss Balancing" (Chen et al., ICML 2018)

    通过平衡各任务梯度幅度来自适应调权。

    Args:
        num_tasks: 任务数量
        alpha: 不对称超参（控制难度平衡强度）
    """

    def __init__(
        self,
        num_tasks: int,
        alpha: float = 1.5,
        init_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.task_weights = nn.Parameter(
            torch.tensor(init_weights or [1.0] * num_tasks, dtype=torch.float32)
        )
        self.register_buffer('initial_losses', torch.zeros(num_tasks))
        self._initialized = False

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        if len(losses) != self.num_tasks:
            raise ValueError(f"损失数量 ({len(losses)}) != num_tasks ({self.num_tasks})")

        if not self._initialized:
            with torch.no_grad():
                self.initial_losses.copy_(
                    torch.tensor([l.item() for l in losses], device=self.task_weights.device)
                )
            self._initialized = True

        nw = self.num_tasks * F.softmax(self.task_weights, dim=0)
        return sum(w * l for w, l in zip(nw, losses))

    def get_weights(self) -> List[float]:
        with torch.no_grad():
            return (self.num_tasks * F.softmax(self.task_weights, dim=0)).tolist()


class FixedWeightMultiTaskLoss(nn.Module):
    """
    固定权重的多任务损失（不可学习）。

    Args:
        weights: 权重列表或 {任务名: 权重} 字典
        normalize: 是否将权重归一化使和为 1
    """

    def __init__(
        self,
        weights: Union[List[float], Dict[str, float]],
        normalize: bool = False,
    ):
        super().__init__()
        if isinstance(weights, dict):
            self.task_names: Optional[List[str]] = list(weights.keys())
            vals = list(weights.values())
        else:
            self.task_names = None
            vals = list(weights)

        if normalize:
            s = sum(vals)
            vals = [v / s for v in vals]

        self.register_buffer('weights', torch.tensor(vals, dtype=torch.float32))
        self.num_tasks = len(vals)

    def forward(
        self, losses: Union[List[torch.Tensor], Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        if isinstance(losses, dict):
            losses = (
                [losses[n] for n in self.task_names]
                if self.task_names
                else list(losses.values())
            )
        return sum(w * l for w, l in zip(self.weights, losses))

    def get_weights(self) -> List[float]:
        return self.weights.tolist()
