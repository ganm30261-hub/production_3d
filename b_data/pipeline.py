"""
流水线编排器（Pipeline）

将解析、转换、预处理串联为可一键执行的完整工作流。

新增论文方法:
    - export_wall_segmentation_data: 导出 FPN 墙体分割训练数据 (image + mask)
    - export_symbol_detection_data:  导出 Faster R-CNN 门窗检测训练数据 (image + bbox)

使用方法：
    from data.pipeline import FloorplanPipeline, PipelineConfig

    pipeline = FloorplanPipeline(PipelineConfig(data_root="/data/cubicasa5k"))
    pipeline.run_full(splits=['train', 'val', 'test'], target_format='yolo')

    # 论文方法
    pipeline.export_wall_segmentation_data(splits=['train', 'val'])
    pipeline.export_symbol_detection_data(splits=['train', 'val'])
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import cv2
import numpy as np

from .cubicasa_parser import CubiCasaParser, ParserConfig
from .coco_converter import COCOConverter, COCOConverterConfig
from .yolo_converter import YOLOConverter, YOLOConverterConfig
from .preprocessing import Preprocessor, PreprocessConfig
from .exceptions import PipelineError, ImageLoadError, SVGParseError

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """流水线总配置"""
    data_root: str = ""

    # 子模块配置
    parser: ParserConfig = field(default_factory=ParserConfig)
    coco: COCOConverterConfig = field(default_factory=COCOConverterConfig)
    yolo: YOLOConverterConfig = field(default_factory=YOLOConverterConfig)
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)

    # 输出目录
    coco_output_dir: str = "./output/coco_annotations"
    yolo_output_dir: str = "./output/yolo_dataset"
    wall_seg_output_dir: str = "./output/wall_segmentation"
    symbol_det_output_dir: str = "./output/symbol_detection"


class FloorplanPipeline:
    """
    一站式流水线编排器。

    典型用法:
        pipeline = FloorplanPipeline(config)
        pipeline.run_full(splits=['train', 'val', 'test'], target_format='yolo')

    也可以单步执行:
        pipeline.step_parse_to_coco(split='train')
        pipeline.step_coco_to_yolo()
    """

    def __init__(self, config: PipelineConfig):
        if not config.data_root:
            raise PipelineError("必须指定 data_root")
        self.config = config
        self.data_root = Path(config.data_root)

        if not self.data_root.exists():
            raise PipelineError(f"数据集根目录不存在: {self.data_root}")

    def run_full(
        self,
        splits: Optional[List[str]] = None,
        target_format: str = 'yolo',
    ) -> Dict:
        """
        执行完整流水线: CubiCasa → COCO → YOLO。

        Args:
            splits: 要处理的划分列表
            target_format: 最终格式 ('coco' 或 'yolo')

        Returns:
            各阶段的输出信息
        """
        splits = splits or ['train', 'val', 'test']
        results: Dict = {'stages': {}}
        t0 = time.time()

        # Step 1: CubiCasa → COCO
        logger.info("=" * 60)
        logger.info("Step 1/2: CubiCasa → COCO")
        logger.info("=" * 60)

        coco_cfg = COCOConverterConfig(
            include_rooms=self.config.coco.include_rooms,
            include_walls=self.config.coco.include_walls,
            include_doors_windows=self.config.coco.include_doors_windows,
            include_structural=self.config.coco.include_structural,
            use_original_size=self.config.coco.use_original_size,
            use_polygon=self.config.coco.use_polygon,
            min_polygon_points=self.config.coco.min_polygon_points,
            min_area=self.config.coco.min_area,
            output_dir=self.config.coco_output_dir,
            enable_resume=self.config.coco.enable_resume,
        )
        coco_converter = COCOConverter(str(self.data_root), coco_cfg)
        coco_results = coco_converter.convert_all_splits(
            splits=splits,
            output_dir=self.config.coco_output_dir,
        )
        results['stages']['coco'] = {
            split: stats.summary() for split, (_, stats) in coco_results.items()
        }

        # Step 2: COCO → YOLO（如果需要）
        if target_format == 'yolo':
            logger.info("=" * 60)
            logger.info("Step 2/2: COCO → YOLO")
            logger.info("=" * 60)

            yolo_converter = YOLOConverter(str(self.data_root), self.config.yolo)
            yolo_results = yolo_converter.convert_all_splits(
                coco_dir=self.config.coco_output_dir,
                output_dir=self.config.yolo_output_dir,
                splits=splits,
            )
            results['stages']['yolo'] = yolo_results

        elapsed = time.time() - t0
        logger.info("=" * 60)
        logger.info("流水线完成，总耗时: %.1f 秒", elapsed)
        logger.info("=" * 60)

        results['elapsed_seconds'] = elapsed
        return results

    def step_parse_to_coco(
        self, split: str = 'train', output_file: Optional[str] = None
    ):
        """单独执行: CubiCasa → COCO。"""
        coco_converter = COCOConverter(str(self.data_root), self.config.coco)
        out = output_file or str(
            Path(self.config.coco_output_dir) / f"coco_annotations_{split}.json"
        )
        return coco_converter.convert(split=split, output_file=out)

    def step_coco_to_yolo(self, coco_dir: Optional[str] = None):
        """单独执行: COCO → YOLO。"""
        yolo_converter = YOLOConverter(str(self.data_root), self.config.yolo)
        return yolo_converter.convert_all_splits(
            coco_dir=coco_dir or self.config.coco_output_dir,
            output_dir=self.config.yolo_output_dir,
        )

    def create_preprocessor(self, mode: str = 'train') -> Preprocessor:
        """创建预处理器实例（用于训练/推理时在线调用）。"""
        return Preprocessor(self.config.preprocessing, mode=mode)

    # ==================================================================
    # 论文方法: 训练数据导出
    # ==================================================================

    def export_wall_segmentation_data(
        self,
        splits: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        crop_to_walls: bool = True,
    ) -> Dict[str, Dict]:
        """
        导出墙体分割训练数据 (FPN 语义分割模型用)。

        论文 Section 2.3 + Section 3:
            输入: 平面图图像
            标签: 墙体二值掩码 (house.walls == 2) * 255
            裁剪: 到墙壁掩码边界，排除未标注区域

        输出结构:
            {output_dir}/{split}/images/000001.png
            {output_dir}/{split}/masks/000001.png
            {output_dir}/filelist_{split}.txt
        """
        splits = splits or ['train', 'val']
        out_root = Path(output_dir or self.config.wall_seg_output_dir)

        parser = CubiCasaParser(str(self.data_root), ParserConfig(
            extract_rooms=False, extract_walls=True,
            extract_doors_windows=False, extract_structural=False,
            use_original_size=self.config.parser.use_original_size,
            split_files=self.config.parser.split_files,
        ))
        preprocessor = self.create_preprocessor(mode='eval') if crop_to_walls else None
        results = {}

        for split in splits:
            logger.info("=" * 60)
            logger.info("导出墙体分割数据: %s", split.upper())
            logger.info("=" * 60)

            img_dir = out_root / split / 'images'
            msk_dir = out_root / split / 'masks'
            img_dir.mkdir(parents=True, exist_ok=True)
            msk_dir.mkdir(parents=True, exist_ok=True)

            samples = parser.load_split_file(split)
            fnames, stats = [], {'total': 0, 'ok': 0, 'skip': 0, 'empty': 0}

            for idx, name in enumerate(samples):
                stats['total'] += 1
                try:
                    s = parser.load_sample(name)
                except (ImageLoadError, SVGParseError):
                    stats['skip'] += 1
                    continue

                if s.wall_mask is None or not s.wall_mask.any():
                    stats['empty'] += 1
                    continue

                image, wall_mask = s.image, s.wall_mask

                # 论文裁剪策略
                if crop_to_walls and preprocessor and s.house_walls is not None:
                    crop = preprocessor.process(image, s.house_walls)
                    image = crop['image']
                    bbox = crop.get('crop_bbox')
                    if bbox:
                        y1, x1, y2, x2 = bbox
                        wall_mask = wall_mask[y1:y2, x1:x2]

                fn = f"{idx:06d}.png"
                cv2.imwrite(str(img_dir / fn), image)
                cv2.imwrite(str(msk_dir / fn), wall_mask)
                fnames.append(fn)
                stats['ok'] += 1

            (out_root / f"filelist_{split}.txt").write_text(
                '\n'.join(fnames) + '\n', encoding='utf-8',
            )
            logger.info("墙体分割 [%s]: %d/%d 成功 (跳过%d, 空%d)",
                         split, stats['ok'], stats['total'], stats['skip'], stats['empty'])
            results[split] = stats

        return results

    def export_symbol_detection_data(
        self,
        splits: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        crop_to_walls: bool = True,
    ) -> Dict[str, Dict]:
        """
        导出门窗检测训练数据 (Faster R-CNN 模型用)。

        论文 Section 2.2:
            输入: 平面图图像
            标签: bbox [x1,y1,x2,y2] + label (1=window, 2=door)
            来源: house.icons → connectedComponentsWithStats → bbox
            裁剪: 到墙壁掩码边界，排除未标注区域

        输出结构:
            {output_dir}/{split}/images/000001.png
            {output_dir}/{split}/annotations.json
        """
        splits = splits or ['train', 'val']
        out_root = Path(output_dir or self.config.symbol_det_output_dir)

        parser = CubiCasaParser(str(self.data_root), ParserConfig(
            extract_rooms=False, extract_walls=False,
            extract_doors_windows=True, extract_structural=False,
            use_original_size=self.config.parser.use_original_size,
            split_files=self.config.parser.split_files,
        ))
        preprocessor = self.create_preprocessor(mode='eval') if crop_to_walls else None
        results = {}

        for split in splits:
            logger.info("=" * 60)
            logger.info("导出门窗检测数据: %s", split.upper())
            logger.info("=" * 60)

            img_dir = out_root / split / 'images'
            img_dir.mkdir(parents=True, exist_ok=True)

            samples = parser.load_split_file(split)
            annotations = []
            stats = {'total': 0, 'ok': 0, 'skip': 0, 'empty': 0,
                     'doors': 0, 'windows': 0}

            for idx, name in enumerate(samples):
                stats['total'] += 1
                try:
                    s = parser.load_sample(name)
                except (ImageLoadError, SVGParseError):
                    stats['skip'] += 1
                    continue

                image = s.image
                doors, windows = s.doors, s.windows

                # 论文裁剪策略 + bbox 坐标调整
                off_x, off_y = 0, 0
                if crop_to_walls and preprocessor and s.house_walls is not None:
                    crop = preprocessor.process(image, s.house_walls, s.house_icons)
                    image = crop['image']
                    bbox = crop.get('crop_bbox')
                    if bbox:
                        off_y, off_x = bbox[0], bbox[1]
                        h, w = image.shape[:2]
                        doors = self._adjust_openings(doors, off_x, off_y, w, h)
                        windows = self._adjust_openings(windows, off_x, off_y, w, h)

                if not doors and not windows:
                    stats['empty'] += 1
                    continue

                fn = f"{idx:06d}.png"
                cv2.imwrite(str(img_dir / fn), image)

                boxes, labels = [], []
                for win in windows:
                    if win.bbox and len(win.bbox) == 4:
                        boxes.append(win.bbox)
                        labels.append(1)
                        stats['windows'] += 1
                for door in doors:
                    if door.bbox and len(door.bbox) == 4:
                        boxes.append(door.bbox)
                        labels.append(2)
                        stats['doors'] += 1

                ih, iw = image.shape[:2]
                annotations.append({
                    'image_id': idx, 'file_name': fn,
                    'width': iw, 'height': ih,
                    'boxes': boxes, 'labels': labels,
                })
                stats['ok'] += 1

            ann_file = out_root / split / 'annotations.json'
            with open(ann_file, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, indent=2, ensure_ascii=False)

            logger.info("门窗检测 [%s]: %d/%d 成功 (门%d 窗%d, 跳过%d, 空%d)",
                         split, stats['ok'], stats['total'],
                         stats['doors'], stats['windows'],
                         stats['skip'], stats['empty'])
            results[split] = stats

        return results

    @staticmethod
    def _adjust_openings(openings, off_x, off_y, img_w, img_h):
        """裁剪后调整门窗 bbox 坐标。"""
        from .schemas import OpeningAnnotation
        out = []
        for op in openings:
            if not op.bbox or len(op.bbox) != 4:
                continue
            x1 = max(0, op.bbox[0] - off_x)
            y1 = max(0, op.bbox[1] - off_y)
            x2 = min(img_w, op.bbox[2] - off_x)
            y2 = min(img_h, op.bbox[3] - off_y)
            if x2 > x1 and y2 > y1:
                out.append(OpeningAnnotation(
                    opening_type=op.opening_type,
                    bbox=[x1, y1, x2, y2],
                    center=[(x1+x2)/2, (y1+y2)/2],
                    area=(x2-x1)*(y2-y1),
                    width=op.width, height=op.height,
                    sill_height=op.sill_height,
                ))
        return out


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description='Floorplan 数据处理流水线')
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--target', type=str, default='yolo',
                    choices=['coco', 'yolo', 'wall_seg', 'symbol_det', 'all_paper'])
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--coco_output', type=str, default='./output/coco_annotations')
    ap.add_argument('--yolo_output', type=str, default='./output/yolo_dataset')
    ap.add_argument('--wall_seg_output', type=str, default='./output/wall_segmentation')
    ap.add_argument('--symbol_det_output', type=str, default='./output/symbol_detection')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--no_crop', action='store_true', help='不执行论文裁剪策略')
    args = ap.parse_args()

    cfg = PipelineConfig(
        data_root=args.data_root,
        coco_output_dir=args.coco_output,
        yolo_output_dir=args.yolo_output,
        wall_seg_output_dir=args.wall_seg_output,
        symbol_det_output_dir=args.symbol_det_output,
    )
    cfg.coco.enable_resume = args.resume
    pipeline = FloorplanPipeline(cfg)
    crop = not args.no_crop

    if args.target in ('coco', 'yolo'):
        pipeline.run_full(splits=args.splits, target_format=args.target)
    elif args.target == 'wall_seg':
        pipeline.export_wall_segmentation_data(splits=args.splits, crop_to_walls=crop)
    elif args.target == 'symbol_det':
        pipeline.export_symbol_detection_data(splits=args.splits, crop_to_walls=crop)
    elif args.target == 'all_paper':
        pipeline.export_wall_segmentation_data(splits=args.splits, crop_to_walls=crop)
        pipeline.export_symbol_detection_data(splits=args.splits, crop_to_walls=crop)
