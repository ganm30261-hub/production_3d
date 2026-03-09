#!/usr/bin/env python3
"""
端到端训练入口脚本

将数据导出和模型训练串联为一键可执行的完整流程。
支持多 GPU (DataParallel)，支持单独执行或全流程执行。

整体流程:
    ┌─────────────────────────────────────────────────────┐
    │  Step 1: 数据导出 (data.pipeline)                     │
    │    CubiCasa5k SVG ──→ YOLO格式 + 分割掩码 + 检测bbox    │
    ├─────────────────────────────────────────────────────┤
    │  Step 2: 模型训练 (models.training_pipeline)           │
    │    2a. 墙体分割: FPN + BCE + Affinity + Kendall        │
    │    2b. 门窗检测: Faster R-CNN + ResNet-50              │
    │    2c. YOLO:     YOLOv8-Seg 实例分割                   │
    └─────────────────────────────────────────────────────┘

使用方法:
    # 全流程: 数据导出 + 三个模型训练
    python main.py --data_root /path/to/cubicasa5k --mode full

    # 仅导出数据
    python main.py --data_root /path/to/cubicasa5k --mode export_only

    # 仅训练（数据已导出）
    python main.py --mode train_only --train wall_seg symbol_det yolo

    # 只训练论文方法的两个模型
    python main.py --mode train_only --train wall_seg symbol_det

    # 只训练墙体分割
    python main.py --mode train_only --train wall_seg --wall_seg_epochs 100

    # 多 GPU 指定
    python main.py --mode train_only --train wall_seg --gpus 0,1
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("main")


# ============================================================================
# 数据导出
# ============================================================================

def run_data_export(args):
    """Step 1: 从 CubiCasa5k 导出所有训练数据。"""
    from b_data.pipeline import FloorplanPipeline, PipelineConfig

    logger.info("=" * 70)
    logger.info("Step 1: 数据导出")
    logger.info("=" * 70)

    cfg = PipelineConfig(
        data_root=args.data_root,
        coco_output_dir=args.coco_output,
        yolo_output_dir=args.yolo_output,
        wall_seg_output_dir=args.wall_seg_output,
        symbol_det_output_dir=args.symbol_det_output,
    )
    pipeline = FloorplanPipeline(cfg)
    splits = args.splits
    crop = not args.no_crop

    results = {}
    t0 = time.time()

    # 1a. YOLO 格式 (原有方案)
    if 'yolo' in args.train:
        logger.info("--- 导出 YOLO 格式 ---")
        results['yolo'] = pipeline.run_full(splits=splits, target_format='yolo')

    # 1b. 墙体分割格式 (论文方法)
    if 'wall_seg' in args.train:
        logger.info("--- 导出墙体分割数据 ---")
        results['wall_seg'] = pipeline.export_wall_segmentation_data(
            splits=splits, crop_to_walls=crop,
        )

    # 1c. 门窗检测格式 (论文方法)
    if 'symbol_det' in args.train:
        logger.info("--- 导出门窗检测数据 ---")
        results['symbol_det'] = pipeline.export_symbol_detection_data(
            splits=splits, crop_to_walls=crop,
        )

    elapsed = time.time() - t0
    logger.info("数据导出完成, 耗时 %.1f 秒", elapsed)
    return results


# ============================================================================
# 模型训练
# ============================================================================

def run_training(args):
    """Step 2: 训练指定的模型。"""
    import torch

    logger.info("=" * 70)
    logger.info("Step 2: 模型训练")
    logger.info("=" * 70)

    # GPU 配置
    gpu_ids = _parse_gpus(args.gpus)
    if gpu_ids:
        logger.info("使用 GPU: %s", gpu_ids)
        torch.cuda.set_device(gpu_ids[0])
        device = f'cuda:{gpu_ids[0]}'
    else:
        device = 'cpu'
        logger.info("使用 CPU")

    results = {}
    t0 = time.time()

    # 2a. 墙体分割
    if 'wall_seg' in args.train:
        logger.info("=" * 50)
        logger.info("训练: 墙体分割 (FPN + ResNet)")
        logger.info("=" * 50)
        results['wall_seg'] = _train_wall_seg(args, device, gpu_ids)

    # 2b. 门窗检测
    if 'symbol_det' in args.train:
        logger.info("=" * 50)
        logger.info("训练: 门窗检测 (Faster R-CNN)")
        logger.info("=" * 50)
        results['symbol_det'] = _train_symbol_det(args, device, gpu_ids)

    # 2c. YOLO
    if 'yolo' in args.train:
        logger.info("=" * 50)
        logger.info("训练: YOLO 实例分割")
        logger.info("=" * 50)
        results['yolo'] = _train_yolo(args, gpu_ids)

    elapsed = time.time() - t0
    logger.info("全部训练完成, 总耗时 %.1f 秒", elapsed)
    return results


def _train_wall_seg(args, device, gpu_ids):
    """训练墙体分割模型，支持多 GPU DataParallel。"""
    from c_models.wall_seg_trainer import WallSegmentationTrainer, WallSegTrainerConfig

    cfg = WallSegTrainerConfig(
        data_dir=args.wall_seg_output,
        backbone=args.backbone,
        pretrained=True,
        target_size=args.img_size,
        num_epochs=args.wall_seg_epochs,
        batch_size=args.wall_seg_batch * max(len(gpu_ids), 1),  # 多 GPU 线性扩展
        lr=args.wall_seg_lr,
        num_workers=args.workers,
        device=device,
        use_affinity=True,
        use_kendall=True,
        save_dir=args.save_dir + '/wall_segmentation',
    )

    trainer = WallSegmentationTrainer(cfg)

    # 多 GPU 支持: 修改 trainer 内部模型为 DataParallel
    if len(gpu_ids) > 1:
        trainer._multi_gpu_ids = gpu_ids
        logger.info("墙体分割: DataParallel on GPUs %s", gpu_ids)

    return trainer.train()


def _train_symbol_det(args, device, gpu_ids):
    """训练门窗检测模型，支持多 GPU DataParallel。"""
    from c_models.symbol_det_trainer import SymbolDetectionTrainer, SymbolDetTrainerConfig

    cfg = SymbolDetTrainerConfig(
        data_dir=args.symbol_det_output,
        pretrained=True,
        num_epochs=args.symbol_det_epochs,
        batch_size=args.symbol_det_batch * max(len(gpu_ids), 1),
        lr=args.symbol_det_lr,
        num_workers=args.workers,
        device=device,
        save_dir=args.save_dir + '/symbol_detection',
    )

    trainer = SymbolDetectionTrainer(cfg)

    if len(gpu_ids) > 1:
        trainer._multi_gpu_ids = gpu_ids
        logger.info("门窗检测: DataParallel on GPUs %s", gpu_ids)

    return trainer.train()


def _train_yolo(args, gpu_ids):
    """训练 YOLO 模型。YOLO 自带多 GPU 支持。"""
    from c_models.yolo_trainer import EnhancedYOLOTrainer, YOLOTrainerConfig

    yaml_path = str(Path(args.yolo_output) / 'dataset.yaml')
    if not Path(yaml_path).exists():
        logger.error("YOLO dataset.yaml 不存在: %s", yaml_path)
        return {'error': f'dataset.yaml not found: {yaml_path}'}

    # YOLO device 格式: "0" 或 "0,1"
    device_str = ','.join(str(g) for g in gpu_ids) if gpu_ids else 'cpu'

    cfg = YOLOTrainerConfig(
        model_name=args.yolo_model,
        epochs=args.yolo_epochs,
        batch_size=args.yolo_batch,
        imgsz=args.img_size,
        device=device_str,
        save_dir=args.save_dir + '/yolo',
    )

    trainer = EnhancedYOLOTrainer(cfg)
    return trainer.train(data_yaml=yaml_path)


# ============================================================================
# 工具函数
# ============================================================================

def _parse_gpus(gpus_str):
    """解析 GPU 列表: '0,1' → [0, 1]"""
    import torch
    if not gpus_str or gpus_str == 'cpu':
        return []
    if gpus_str == 'auto':
        n = torch.cuda.device_count()
        return list(range(n)) if n > 0 else []
    return [int(g.strip()) for g in gpus_str.split(',')]


def print_summary(args):
    """打印运行配置摘要。"""
    logger.info("=" * 70)
    logger.info("运行配置")
    logger.info("=" * 70)
    logger.info("  模式:       %s", args.mode)
    logger.info("  训练模型:    %s", args.train)
    logger.info("  GPU:        %s", args.gpus)
    logger.info("  数据集:     %s", args.data_root or '(已导出)')
    if 'wall_seg' in args.train:
        logger.info("  墙体分割:   epochs=%d, batch=%d, lr=%.1e, backbone=%s",
                     args.wall_seg_epochs, args.wall_seg_batch,
                     args.wall_seg_lr, args.backbone)
    if 'symbol_det' in args.train:
        logger.info("  门窗检测:   epochs=%d, batch=%d, lr=%.1e",
                     args.symbol_det_epochs, args.symbol_det_batch,
                     args.symbol_det_lr)
    if 'yolo' in args.train:
        logger.info("  YOLO:       epochs=%d, batch=%d, model=%s",
                     args.yolo_epochs, args.yolo_batch, args.yolo_model)
    logger.info("  输出目录:   %s", args.save_dir)
    logger.info("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='CubiCasa5k 平面图识别 — 端到端训练入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 全流程
  python main.py --data_root /data/cubicasa5k --mode full --gpus 0,1

  # 仅导出数据
  python main.py --data_root /data/cubicasa5k --mode export_only

  # 仅训练论文方法
  python main.py --mode train_only --train wall_seg symbol_det --gpus 0,1

  # 仅训练墙体分割，100 epochs
  python main.py --mode train_only --train wall_seg --wall_seg_epochs 100
        """,
    )

    # 运行模式
    p.add_argument('--mode', type=str, default='full',
                   choices=['full', 'export_only', 'train_only'],
                   help='full=导出+训练, export_only=仅导出, train_only=仅训练')
    p.add_argument('--train', nargs='+', default=['wall_seg', 'symbol_det', 'yolo'],
                   choices=['wall_seg', 'symbol_det', 'yolo'],
                   help='要训练的模型列表')

    # 数据路径
    p.add_argument('--data_root', type=str, default='',
                   help='CubiCasa5k 数据集根目录')
    p.add_argument('--splits', nargs='+', default=['train', 'val'],
                   help='数据划分')

    # 数据输出目录
    p.add_argument('--wall_seg_output', type=str, default='./output/wall_segmentation')
    p.add_argument('--symbol_det_output', type=str, default='./output/symbol_detection')
    p.add_argument('--yolo_output', type=str, default='./output/yolo_dataset')
    p.add_argument('--coco_output', type=str, default='./output/coco_annotations')
    p.add_argument('--no_crop', action='store_true',
                   help='不执行论文裁剪策略')

    # GPU
    p.add_argument('--gpus', type=str, default='auto',
                   help='GPU 设备: auto/0/0,1/cpu')

    # 通用训练参数
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='./runs')
    p.add_argument('--backbone', type=str, default='resnet50',
                   choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])

    # 墙体分割参数
    p.add_argument('--wall_seg_epochs', type=int, default=50)
    p.add_argument('--wall_seg_batch', type=int, default=4,
                   help='单 GPU batch size，多 GPU 自动乘以 GPU 数')
    p.add_argument('--wall_seg_lr', type=float, default=1e-4)

    # 门窗检测参数
    p.add_argument('--symbol_det_epochs', type=int, default=15)
    p.add_argument('--symbol_det_batch', type=int, default=2,
                   help='单 GPU batch size')
    p.add_argument('--symbol_det_lr', type=float, default=5e-3)

    # YOLO 参数
    p.add_argument('--yolo_model', type=str, default='yolov8m-seg.pt')
    p.add_argument('--yolo_epochs', type=int, default=100)
    p.add_argument('--yolo_batch', type=int, default=8)

    return p.parse_args()


def main():
    args = parse_args()

    # 日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(args.save_dir) / 'train.log', mode='a'),
        ],
    )
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # 验证参数
    if args.mode in ('full', 'export_only') and not args.data_root:
        logger.error("--mode %s 需要指定 --data_root", args.mode)
        sys.exit(1)

    print_summary(args)
    total_t0 = time.time()

    # Step 1: 数据导出
    if args.mode in ('full', 'export_only'):
        run_data_export(args)

    # Step 2: 模型训练
    if args.mode in ('full', 'train_only'):
        run_training(args)

    logger.info("=" * 70)
    logger.info("全部完成, 总耗时 %.1f 秒", time.time() - total_t0)
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
