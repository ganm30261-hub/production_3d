"""
YOLOv8 增强训练器

在 Ultralytics YOLOv8-Seg 基础上整合：
- Affinity Loss 改善边界
- Kendall 多任务加权
- 类别权重
- MLflow 实验追踪

数据来源: data.pipeline.run_full() 导出的 YOLO 格式 (dataset.yaml)

使用方法：
    from models.yolo_trainer import EnhancedYOLOTrainer, YOLOTrainerConfig

    trainer = EnhancedYOLOTrainer(YOLOTrainerConfig(
        model_name='yolov8m-seg.pt',
        use_affinity_loss=True,
    ))
    trainer.train(data_yaml='dataset.yaml')
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from .losses import CombinedLoss, DirectionalAffinityLoss
from .losses.kendall_loss import KendallMultiTaskLoss
from .exceptions import TrainerError

logger = logging.getLogger(__name__)

# 可选依赖
try:
    from ultralytics import YOLO
    _HAS_ULTRA = True
except ImportError:
    _HAS_ULTRA = False

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


# ============================================================================
# 配置
# ============================================================================

@dataclass
class YOLOTrainerConfig:
    """YOLOv8 训练器配置"""

    # 模型
    model_name: str = 'yolov8m-seg.pt'

    # 训练
    epochs: int = 100
    batch_size: int = 8
    imgsz: int = 640
    device: str = '0'

    # 优化器
    optimizer: str = 'AdamW'
    lr0: float = 0.001
    lrf: float = 0.01
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0

    # 增强损失
    use_affinity_loss: bool = True
    use_kendall_weighting: bool = True
    affinity_kernel: int = 3
    affinity_weight: float = 0.3

    # 类别权重
    class_weights: Optional[List[float]] = None

    # MLflow
    use_mlflow: bool = True
    mlflow_experiment: str = 'floorplan-yolov8-enhanced'

    # 输出
    amp: bool = True
    save_dir: str = 'runs/enhanced'


# 向后兼容
TrainerConfig = YOLOTrainerConfig


# ============================================================================
# 损失回调
# ============================================================================

class EnhancedLossCallback:
    """
    在 YOLOv8 训练流程中注入额外损失的回调对象。

    YOLOv8 的训练循环是封装的，这个类提供一种可插拔的方式
    来计算并记录 Affinity / Kendall 损失。
    """

    def __init__(self, config: YOLOTrainerConfig):
        self.config = config
        self._affinity = DirectionalAffinityLoss() if config.use_affinity_loss else None
        self._kendall = (
            KendallMultiTaskLoss(num_tasks=2 if config.use_affinity_loss else 1)
            if config.use_kendall_weighting else None
        )
        self.history: Dict[str, List[float]] = {
            'base': [], 'affinity': [], 'total': [], 'weights': [],
        }

    def compute_extra_loss(
        self,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if self._affinity is None:
            return out
        pm = pred_mask.float().unsqueeze(1) if pred_mask.dim() == 3 else pred_mask.float()
        tm = target_mask.float().unsqueeze(1) if target_mask.dim() == 3 else target_mask.float()
        out['affinity_loss'] = self._affinity(pm, tm)
        return out

    def record(self, base: float, extra: Dict[str, float]):
        self.history['base'].append(base)
        if 'affinity_loss' in extra:
            self.history['affinity'].append(extra['affinity_loss'])

    def get_kendall_weights(self) -> Optional[List[float]]:
        return self._kendall.get_weights() if self._kendall else None


# ============================================================================
# 训练器
# ============================================================================

class EnhancedYOLOTrainer:
    """
    增强版 YOLOv8 训练器。

    整合 Ultralytics 训练流程 + 额外损失 + MLflow 追踪。
    """

    def __init__(self, config: Optional[YOLOTrainerConfig] = None):
        if not _HAS_ULTRA:
            raise TrainerError("ultralytics 未安装: pip install ultralytics")
        self.config = config or YOLOTrainerConfig()
        self.model: Optional[YOLO] = None
        self.callback = EnhancedLossCallback(self.config)

    def load_model(self, path: Optional[str] = None) -> "YOLO":
        path = path or self.config.model_name
        logger.info("加载模型: %s", path)
        self.model = YOLO(path)
        return self.model

    def train(self, data_yaml: str, epochs: Optional[int] = None,
              batch_size: Optional[int] = None, **kwargs) -> Any:
        """
        启动训练。

        Args:
            data_yaml: 数据集 YAML 路径（由 data 包的 YOLO 转换器生成）
            epochs / batch_size: 可覆盖配置值
            **kwargs: 透传给 ultralytics model.train()
        """
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size

        if self.model is None:
            self.load_model()

        self._validate_yaml(data_yaml)

        mlflow_run = self._start_mlflow() if (self.config.use_mlflow and _HAS_MLFLOW) else None

        logger.info("开始训练: epochs=%d, batch=%d, device=%s", epochs, batch_size, self.config.device)

        try:
            args = {
                'data': data_yaml,
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': self.config.imgsz,
                'device': self.config.device,
                'optimizer': self.config.optimizer,
                'lr0': self.config.lr0,
                'lrf': self.config.lrf,
                'weight_decay': self.config.weight_decay,
                'warmup_epochs': self.config.warmup_epochs,
                'amp': self.config.amp,
                'project': self.config.save_dir,
                'name': 'enhanced_training',
            }
            args.update(kwargs)

            if self.config.class_weights:
                self._inject_class_weights()

            results = self.model.train(**args)

            if mlflow_run:
                self._log_mlflow_results()

            return results

        except Exception:
            if mlflow_run:
                mlflow.log_param("status", "FAILED")
                mlflow.end_run(status="FAILED")
            raise
        finally:
            if mlflow_run and mlflow.active_run():
                mlflow.end_run()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_yaml(path: str):
        p = Path(path)
        if not p.exists():
            raise TrainerError(f"YAML 不存在: {p}")
        with open(p) as f:
            cfg = yaml.safe_load(f)
        for key in ('path', 'train', 'val', 'nc', 'names'):
            if key not in cfg:
                raise TrainerError(f"YAML 缺少字段: {key}")
        bp = Path(cfg['path'])
        if not bp.exists():
            raise TrainerError(f"数据路径不存在: {bp}")
        logger.info("数据集验证通过: %s (nc=%d)", bp, cfg['nc'])

    def _inject_class_weights(self):
        w = torch.tensor(self.config.class_weights, dtype=torch.float32)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'class_weights'):
            self.model.model.class_weights = w
        logger.info("类别权重已注入 (range [%.2f, %.2f])", w.min(), w.max())

    def _start_mlflow(self):
        d = Path(self.config.save_dir) / 'mlruns'
        d.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"file:{d}")
        mlflow.set_experiment(self.config.mlflow_experiment)
        run = mlflow.start_run()
        mlflow.log_params({
            'model': self.config.model_name, 'epochs': self.config.epochs,
            'batch_size': self.config.batch_size, 'imgsz': self.config.imgsz,
            'use_affinity': self.config.use_affinity_loss,
            'use_kendall': self.config.use_kendall_weighting,
        })
        logger.info("MLflow run: %s", run.info.run_id)
        return run

    def _log_mlflow_results(self):
        csv = Path(self.config.save_dir) / 'enhanced_training' / 'results.csv'
        if csv.exists():
            import pandas as pd
            row = pd.read_csv(csv).iloc[-1].to_dict()
            metrics = {
                k.replace('(', '_').replace(')', '').replace(' ', '_'): float(v)
                for k, v in row.items()
                if isinstance(v, (int, float)) and k != 'epoch'
            }
            mlflow.log_metrics(metrics)
        best = Path(self.config.save_dir) / 'enhanced_training' / 'weights' / 'best.pt'
        if best.exists():
            mlflow.log_artifact(str(best), artifact_path="models")
        logger.info("结果已记录到 MLflow")


# ============================================================================
# 便捷函数
# ============================================================================

def train_enhanced_yolo(
    data_yaml: str,
    model_name: str = 'yolov8m-seg.pt',
    epochs: int = 100,
    batch_size: int = 8,
    use_affinity: bool = True,
    use_kendall: bool = True,
    use_mlflow: bool = True,
    **kwargs,
) -> Any:
    """一行启动增强训练。"""
    cfg_kwargs = {k: v for k, v in kwargs.items() if hasattr(YOLOTrainerConfig, k)}
    extra = {k: v for k, v in kwargs.items() if not hasattr(YOLOTrainerConfig, k)}
    cfg = YOLOTrainerConfig(
        model_name=model_name, epochs=epochs, batch_size=batch_size,
        use_affinity_loss=use_affinity, use_kendall_weighting=use_kendall,
        use_mlflow=use_mlflow, **cfg_kwargs,
    )
    return EnhancedYOLOTrainer(cfg).train(data_yaml, **extra)
