"""
训练流水线编排器

将三个独立的训练器串联为统一的训练入口，
支持单独执行或按顺序批量执行。

三种训练策略:
    yolo       — YOLOv8 实例分割 (原有方案, 所有类别)
    wall_seg   — FPN 墙体语义分割 (论文 Section 2.3)
    symbol_det — Faster R-CNN 门窗检测 (论文 Section 2.2)

使用方法:
    from models.training_pipeline import TrainingPipeline, TrainingPipelineConfig

    # 一键执行论文方法的两个模型训练
    pipeline = TrainingPipeline(TrainingPipelineConfig(
        wall_seg=WallSegTrainerConfig(data_dir='output/wall_segmentation'),
        symbol_det=SymbolDetTrainerConfig(data_dir='output/symbol_detection'),
    ))
    results = pipeline.run(strategies=['wall_seg', 'symbol_det'])

    # 执行全部三种训练
    results = pipeline.run(strategies=['yolo', 'wall_seg', 'symbol_det'],
                           yolo_data_yaml='output/yolo_dataset/dataset.yaml')

CLI:
    python -m models.training_pipeline --strategy wall_seg --data_dir output/wall_segmentation
    python -m models.training_pipeline --strategy symbol_det --data_dir output/symbol_detection
    python -m models.training_pipeline --strategy all --yolo_yaml output/yolo_dataset/dataset.yaml
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .yolo_trainer import EnhancedYOLOTrainer, YOLOTrainerConfig
from .wall_seg_trainer import WallSegmentationTrainer, WallSegTrainerConfig
from .symbol_det_trainer import SymbolDetectionTrainer, SymbolDetTrainerConfig
from .exceptions import TrainerError

logger = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class TrainingPipelineConfig:
    """训练流水线总配置"""
    yolo: YOLOTrainerConfig = field(default_factory=YOLOTrainerConfig)
    wall_seg: WallSegTrainerConfig = field(default_factory=WallSegTrainerConfig)
    symbol_det: SymbolDetTrainerConfig = field(default_factory=SymbolDetTrainerConfig)


# ============================================================================
# 流水线
# ============================================================================

class TrainingPipeline:
    """
    训练流水线编排器。

    管理三个训练器的生命周期，支持:
    - 单独执行某个策略
    - 按顺序批量执行多个策略
    - 统一收集训练结果

    与 data.pipeline 的关系:
        data.pipeline.run_full()                      → yolo_trainer 使用
        data.pipeline.export_wall_segmentation_data()  → wall_seg_trainer 使用
        data.pipeline.export_symbol_detection_data()   → symbol_det_trainer 使用
    """

    VALID_STRATEGIES = {'yolo', 'wall_seg', 'symbol_det'}

    def __init__(self, config: Optional[TrainingPipelineConfig] = None):
        self.config = config or TrainingPipelineConfig()

    def run(
        self,
        strategies: Optional[List[str]] = None,
        yolo_data_yaml: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行指定的训练策略。

        Args:
            strategies: 要执行的策略列表，如 ['wall_seg', 'symbol_det']
                       None 时默认执行论文方法的两个模型
                       'all' 可用作快捷方式执行全部三个
            yolo_data_yaml: YOLO 训练所需的 dataset.yaml 路径
                           仅当 strategies 包含 'yolo' 时需要

        Returns:
            {strategy_name: result_dict, ..., 'elapsed': seconds}
        """
        if strategies is None:
            strategies = ['wall_seg', 'symbol_det']
        elif strategies == ['all']:
            strategies = ['yolo', 'wall_seg', 'symbol_det']

        # 验证
        for s in strategies:
            if s not in self.VALID_STRATEGIES:
                raise TrainerError(
                    f"未知策略 '{s}'，可选: {self.VALID_STRATEGIES}"
                )
        if 'yolo' in strategies and not yolo_data_yaml:
            raise TrainerError("策略包含 'yolo' 时必须指定 yolo_data_yaml")

        results: Dict[str, Any] = {}
        t0 = time.time()

        for strategy in strategies:
            logger.info("=" * 60)
            logger.info("训练策略: %s", strategy.upper())
            logger.info("=" * 60)

            if strategy == 'yolo':
                results['yolo'] = self._run_yolo(yolo_data_yaml)
            elif strategy == 'wall_seg':
                results['wall_seg'] = self._run_wall_seg()
            elif strategy == 'symbol_det':
                results['symbol_det'] = self._run_symbol_det()

        results['elapsed'] = time.time() - t0
        logger.info("=" * 60)
        logger.info("训练流水线完成, 耗时 %.1f 秒", results['elapsed'])
        logger.info("=" * 60)

        return results

    def run_yolo(self, data_yaml: str, **kwargs) -> Any:
        """单独执行 YOLO 训练。"""
        return self._run_yolo(data_yaml, **kwargs)

    def run_wall_seg(self) -> Dict[str, Any]:
        """单独执行墙体分割训练。"""
        return self._run_wall_seg()

    def run_symbol_det(self) -> Dict[str, Any]:
        """单独执行门窗检测训练。"""
        return self._run_symbol_det()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _run_yolo(self, data_yaml, **kwargs):
        logger.info("启动 YOLO 训练...")
        trainer = EnhancedYOLOTrainer(self.config.yolo)
        return trainer.train(data_yaml=data_yaml, **kwargs)

    def _run_wall_seg(self):
        logger.info("启动墙体分割训练...")
        trainer = WallSegmentationTrainer(self.config.wall_seg)
        return trainer.train()

    def _run_symbol_det(self):
        logger.info("启动门窗检测训练...")
        trainer = SymbolDetectionTrainer(self.config.symbol_det)
        return trainer.train()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description='模型训练流水线')
    ap.add_argument('--strategy', type=str, default='all',
                    choices=['yolo', 'wall_seg', 'symbol_det', 'paper', 'all'],
                    help='训练策略: yolo/wall_seg/symbol_det/paper(两个论文模型)/all(全部三个)')
    ap.add_argument('--yolo_yaml', type=str, default=None,
                    help='YOLO dataset.yaml 路径')

    # 墙体分割参数
    ap.add_argument('--wall_seg_data', type=str, default='./output/wall_segmentation')
    ap.add_argument('--wall_seg_epochs', type=int, default=50)
    ap.add_argument('--wall_seg_batch', type=int, default=4)
    ap.add_argument('--wall_seg_backbone', type=str, default='resnet50')

    # 门窗检测参数
    ap.add_argument('--symbol_det_data', type=str, default='./output/symbol_detection')
    ap.add_argument('--symbol_det_epochs', type=int, default=15)
    ap.add_argument('--symbol_det_batch', type=int, default=2)

    # 通用
    ap.add_argument('--device', type=str, default='auto')

    args = ap.parse_args()

    # 构建配置
    cfg = TrainingPipelineConfig(
        wall_seg=WallSegTrainerConfig(
            data_dir=args.wall_seg_data,
            num_epochs=args.wall_seg_epochs,
            batch_size=args.wall_seg_batch,
            backbone=args.wall_seg_backbone,
            device=args.device,
        ),
        symbol_det=SymbolDetTrainerConfig(
            data_dir=args.symbol_det_data,
            num_epochs=args.symbol_det_epochs,
            batch_size=args.symbol_det_batch,
            device=args.device,
        ),
    )

    pipeline = TrainingPipeline(cfg)

    # 映射策略
    strategy_map = {
        'yolo': ['yolo'],
        'wall_seg': ['wall_seg'],
        'symbol_det': ['symbol_det'],
        'paper': ['wall_seg', 'symbol_det'],
        'all': ['yolo', 'wall_seg', 'symbol_det'],
    }

    strategies = strategy_map[args.strategy]
    results = pipeline.run(strategies=strategies, yolo_data_yaml=args.yolo_yaml)

    # 打印结果摘要
    for name, result in results.items():
        if name == 'elapsed':
            continue
        if isinstance(result, dict):
            logger.info("结果 [%s]: %s",
                         name, {k: v for k, v in result.items() if k != 'history'})
