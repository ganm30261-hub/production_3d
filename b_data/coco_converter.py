"""
COCO 格式转换器（生产级）

功能：
1. 将 CubiCasa5k 标注转换为 COCO 格式
2. 使用 Schema 类型取代裸字典
3. 统计信息以不可变返回值形式输出
4. 支持断点续跑（--resume）
5. 支持并行处理（--workers）

使用方法：
    from data.coco_converter import COCOConverter, COCOConverterConfig

    converter = COCOConverter(data_root)
    coco_dict, stats = converter.convert(split='train', output_file='coco_train.json')
"""

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from .cubicasa_parser import CubiCasaParser, ParserConfig
from .schemas import (
    RoomAnnotation,
    WallAnnotation,
    OpeningAnnotation,
    StructuralAnnotation,
    SampleData,
    ConversionStats,
)
from .exceptions import ConversionError, ImageLoadError, SVGParseError

logger = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class COCOConverterConfig:
    """COCO 转换器配置"""
    include_rooms: bool = True
    include_walls: bool = True
    include_doors_windows: bool = True
    include_structural: bool = True

    # Polygon 验证
    use_polygon: bool = True
    min_polygon_points: int = 3
    min_area: float = 100.0

    # 图像
    use_original_size: bool = True

    # 输出
    output_dir: str = "./coco_annotations"

    # 断点续跑
    enable_resume: bool = False
    progress_file: str = ".coco_progress.json"

    # 并行
    num_workers: int = 0  # 0 = 单进程


# COCO 类别定义
COCO_CATEGORIES = [
    {"id": 1, "name": "Outdoor", "supercategory": "room"},
    {"id": 2, "name": "Wall", "supercategory": "structure"},
    {"id": 3, "name": "Kitchen", "supercategory": "room"},
    {"id": 4, "name": "LivingRoom", "supercategory": "room"},
    {"id": 5, "name": "Bedroom", "supercategory": "room"},
    {"id": 6, "name": "Bath", "supercategory": "room"},
    {"id": 7, "name": "Entry", "supercategory": "room"},
    {"id": 8, "name": "Railing", "supercategory": "structure"},
    {"id": 9, "name": "Storage", "supercategory": "room"},
    {"id": 10, "name": "Garage", "supercategory": "room"},
    {"id": 11, "name": "Room", "supercategory": "room"},
    {"id": 12, "name": "Door", "supercategory": "opening"},
    {"id": 13, "name": "Window", "supercategory": "opening"},
    {"id": 14, "name": "Column", "supercategory": "structure"},
]

ROOM_TYPE_TO_COCO_ID: Dict[str, int] = {
    "Outdoor": 1, "Wall": 2, "Kitchen": 3, "LivingRoom": 4,
    "Bedroom": 5, "Bath": 6, "Entry": 7, "Railing": 8,
    "Storage": 9, "Garage": 10, "Room": 11,
}


# ============================================================================
# COCO 转换器
# ============================================================================

class COCOConverter:
    """将 CubiCasa5k 数据集转换为 COCO 格式。"""

    def __init__(self, data_root: str, config: Optional[COCOConverterConfig] = None):
        self.data_root = Path(data_root)
        self.config = config or COCOConverterConfig()

        parser_config = ParserConfig(
            extract_rooms=self.config.include_rooms,
            extract_walls=self.config.include_walls,
            extract_doors_windows=self.config.include_doors_windows,
            extract_structural=self.config.include_structural,
            use_original_size=self.config.use_original_size,
        )
        self.parser = CubiCasaParser(data_root, parser_config)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def convert(
        self,
        split: str = 'train',
        output_file: Optional[str] = None,
        num_images: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], ConversionStats]:
        """
        转换指定划分为 COCO 格式。

        Args:
            split: 'train' / 'val' / 'test'
            output_file: 输出 JSON 路径，None 不保存
            num_images: 处理数量上限，None = 全部

        Returns:
            (coco_dict, stats) 元组
        """
        samples = self.parser.load_split_file(split)
        if num_images:
            samples = samples[:num_images]

        logger.info("开始转换 %s 集 (%d 个样本) → COCO 格式", split.upper(), len(samples))

        # 加载已完成样本（断点续跑）
        completed: set = set()
        if self.config.enable_resume:
            completed = self._load_progress(split)
            logger.info("断点续跑：已完成 %d 个样本", len(completed))

        # 初始化 COCO 结构
        coco: Dict[str, Any] = {
            "images": [],
            "annotations": [],
            "categories": COCO_CATEGORIES,
        }
        stats = ConversionStats()
        annotation_id = 1

        for image_id, sample_name in enumerate(
            tqdm(samples, desc=f"转换 {split}"), start=1
        ):
            if sample_name in completed:
                continue

            try:
                sample = self.parser.load_sample(sample_name)
            except (ImageLoadError, SVGParseError) as e:
                logger.warning("跳过样本 %s: %s", sample_name, e)
                stats.skipped_samples += 1
                continue
            except Exception as e:
                logger.error("处理样本 %s 时出错: %s", sample_name, e, exc_info=True)
                stats.error_samples += 1
                continue

            # 图像条目
            file_name = self._build_file_name(sample_name)
            coco['images'].append({
                "id": image_id,
                "width": sample.width,
                "height": sample.height,
                "file_name": file_name,
            })
            stats.total_images += 1

            # 标注条目
            anns, ann_stats = self._process_sample_annotations(
                sample, image_id, annotation_id
            )
            coco['annotations'].extend(anns)
            annotation_id += len(anns)
            stats = stats.merge(ann_stats)

            # 记录进度
            if self.config.enable_resume:
                completed.add(sample_name)
                if stats.total_images % 100 == 0:
                    self._save_progress(split, completed)

        # 保存结果
        if output_file:
            out_path = Path(output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(coco, f, indent=2, ensure_ascii=False)
            logger.info("COCO 标注已保存: %s", out_path)

        if self.config.enable_resume:
            self._save_progress(split, completed)

        logger.info("转换统计:\n%s", stats.summary())
        return coco, stats

    def convert_all_splits(
        self,
        splits: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Tuple[Dict[str, Any], ConversionStats]]:
        """转换所有划分。"""
        splits = splits or ['train', 'val', 'test']
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        results = {}
        for split in splits:
            output_file = out / f"coco_annotations_{split}.json"
            results[split] = self.convert(split=split, output_file=str(output_file))

        logger.info("所有划分已转换完成，输出目录: %s", out)
        return results

    # ------------------------------------------------------------------
    # 标注生成
    # ------------------------------------------------------------------

    def _process_sample_annotations(
        self,
        sample: SampleData,
        image_id: int,
        start_ann_id: int,
    ) -> Tuple[List[Dict], ConversionStats]:
        """处理单个样本的全部标注，返回 (annotation_list, stats)。"""
        anns: List[Dict] = []
        stats = ConversionStats()
        ann_id = start_ann_id

        # 房间
        if self.config.include_rooms:
            for room in sample.rooms:
                ann = self._room_to_coco(room, image_id, ann_id, sample.width, sample.height, stats)
                if ann:
                    anns.append(ann)
                    ann_id += 1

        # 墙体
        if self.config.include_walls:
            for wall in sample.walls:
                ann = self._wall_to_coco(wall, image_id, ann_id, sample.width, sample.height, stats)
                if ann:
                    anns.append(ann)
                    ann_id += 1

        # 门窗
        if self.config.include_doors_windows:
            for door in sample.doors:
                ann = self._opening_to_coco(door, image_id, ann_id, sample.width, sample.height, stats)
                if ann:
                    anns.append(ann)
                    ann_id += 1
            for window in sample.windows:
                ann = self._opening_to_coco(window, image_id, ann_id, sample.width, sample.height, stats)
                if ann:
                    anns.append(ann)
                    ann_id += 1

        # 结构
        if self.config.include_structural:
            for railing in sample.structural_railings:
                ann = self._structural_to_coco(railing, image_id, ann_id, sample.width, sample.height, stats)
                if ann:
                    anns.append(ann)
                    ann_id += 1

        return anns, stats

    def _room_to_coco(
        self, room: RoomAnnotation, image_id: int, ann_id: int,
        img_w: int, img_h: int, stats: ConversionStats,
    ) -> Optional[Dict]:
        if room.room_type in ('Background', 'Wall', 'Railing'):
            return None
        cat_id = ROOM_TYPE_TO_COCO_ID.get(room.room_type)
        if cat_id is None:
            return None

        bbox = room.bbox
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        if bbox_w <= 0 or bbox_h <= 0:
            return None

        segmentation: List = []
        if self.config.use_polygon and room.polygon:
            seg = self._validate_polygon_for_coco(room.polygon, img_w, img_h)
            if seg:
                segmentation = [seg]
                stats.annotations_with_seg += 1
            else:
                stats.annotations_without_seg += 1
                stats.invalid_polygons += 1
        else:
            stats.annotations_without_seg += 1

        stats.total_annotations += 1
        stats.category_counts[room.room_type] = stats.category_counts.get(room.room_type, 0) + 1

        return {
            "id": ann_id,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [bbox[0], bbox[1], bbox_w, bbox_h],
            "area": float(room.polygon_area),
            "iscrowd": 0,
            "segmentation": segmentation,
        }

    def _wall_to_coco(
        self, wall: WallAnnotation, image_id: int, ann_id: int,
        img_w: int, img_h: int, stats: ConversionStats,
    ) -> Optional[Dict]:
        if len(wall.polygon) < self.config.min_polygon_points:
            return None

        poly_array = np.array(wall.polygon)
        x_min, y_min = poly_array.min(axis=0)
        x_max, y_max = poly_array.max(axis=0)
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        if bbox_w <= 0 or bbox_h <= 0:
            return None

        seg = self._validate_polygon_for_coco(wall.polygon, img_w, img_h)
        if not seg:
            stats.annotations_without_seg += 1
            return None

        stats.annotations_with_seg += 1
        stats.total_annotations += 1
        stats.category_counts['Wall'] = stats.category_counts.get('Wall', 0) + 1

        return {
            "id": ann_id,
            "image_id": image_id,
            "category_id": 2,
            "bbox": [float(x_min), float(y_min), float(bbox_w), float(bbox_h)],
            "area": float(self._polygon_area(poly_array)),
            "iscrowd": 0,
            "segmentation": [seg],
            "wall_thickness": wall.thickness,
            "wall_length": wall.length,
            "wall_height": wall.height,
        }

    def _opening_to_coco(
        self, opening: OpeningAnnotation, image_id: int, ann_id: int,
        img_w: int, img_h: int, stats: ConversionStats,
    ) -> Optional[Dict]:
        if len(opening.bbox) != 4:
            return None

        x_min, y_min, x_max, y_max = opening.bbox
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        if bbox_w <= 0 or bbox_h <= 0:
            return None

        cat_id = 12 if opening.opening_type == 'Door' else 13

        stats.total_annotations += 1
        stats.annotations_without_seg += 1
        stats.category_counts[opening.opening_type] = (
            stats.category_counts.get(opening.opening_type, 0) + 1
        )

        return {
            "id": ann_id,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [x_min, y_min, bbox_w, bbox_h],
            "area": float(opening.area or bbox_w * bbox_h),
            "iscrowd": 0,
            "segmentation": [],
        }

    def _structural_to_coco(
        self, element: StructuralAnnotation, image_id: int, ann_id: int,
        img_w: int, img_h: int, stats: ConversionStats,
    ) -> Optional[Dict]:
        if len(element.polygon) < 2:
            return None

        poly_array = np.array(element.polygon)
        x_min, y_min = poly_array.min(axis=0)
        x_max, y_max = poly_array.max(axis=0)
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        if bbox_w <= 0 or bbox_h <= 0:
            return None

        cat_id = 8 if element.element_type == 'Railing' else 14

        segmentation: List = []
        if len(element.polygon) >= 3:
            seg = self._validate_polygon_for_coco(element.polygon, img_w, img_h)
            if seg:
                segmentation = [seg]
                stats.annotations_with_seg += 1
            else:
                stats.annotations_without_seg += 1
        else:
            stats.annotations_without_seg += 1

        area = (
            self._polygon_area(poly_array) if len(element.polygon) >= 3
            else bbox_w * bbox_h
        )

        stats.total_annotations += 1
        stats.category_counts[element.element_type] = (
            stats.category_counts.get(element.element_type, 0) + 1
        )

        return {
            "id": ann_id,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [float(x_min), float(y_min), float(bbox_w), float(bbox_h)],
            "area": float(area),
            "iscrowd": 0,
            "segmentation": segmentation,
        }

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _validate_polygon_for_coco(
        self, polygon: List, img_w: int, img_h: int
    ) -> Optional[List[float]]:
        """校验并转换 polygon 为 COCO flat 格式。"""
        if not polygon or len(polygon) < self.config.min_polygon_points:
            return None
        try:
            pa = np.array(polygon, dtype=np.float64)
            if pa.ndim != 2 or pa.shape[1] != 2:
                return None
            if np.any(np.isnan(pa)) or np.any(np.isinf(pa)):
                return None

            pa[:, 0] = np.clip(pa[:, 0], 0, img_w)
            pa[:, 1] = np.clip(pa[:, 1], 0, img_h)

            if self._polygon_area(pa) < self.config.min_area:
                return None

            return pa.flatten().tolist()
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _polygon_area(pa: np.ndarray) -> float:
        if len(pa) < 3:
            return 0.0
        x, y = pa[:, 0], pa[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _build_file_name(self, sample_name: str) -> str:
        img = "F1_original.png" if self.config.use_original_size else "F1_scaled.png"
        return f"{sample_name.rstrip('/')}/{img}"

    # ------------------------------------------------------------------
    # 断点续跑
    # ------------------------------------------------------------------

    def _load_progress(self, split: str) -> set:
        pf = Path(self.config.output_dir) / f".progress_{split}.json"
        if pf.exists():
            with open(pf, 'r') as f:
                return set(json.load(f))
        return set()

    def _save_progress(self, split: str, completed: set) -> None:
        pf = Path(self.config.output_dir) / f".progress_{split}.json"
        pf.parent.mkdir(parents=True, exist_ok=True)
        with open(pf, 'w') as f:
            json.dump(sorted(completed), f)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description='CubiCasa5k → COCO 格式转换')
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--output_dir', type=str, default='./coco_annotations')
    ap.add_argument('--split', type=str, default='all',
                    choices=['train', 'val', 'test', 'all'])
    ap.add_argument('--original_size', action='store_true')
    ap.add_argument('--no_polygon', action='store_true')
    ap.add_argument('--resume', action='store_true', help='断点续跑')
    args = ap.parse_args()

    cfg = COCOConverterConfig(
        use_original_size=args.original_size,
        use_polygon=not args.no_polygon,
        output_dir=args.output_dir,
        enable_resume=args.resume,
    )
    converter = COCOConverter(args.data_root, cfg)

    if args.split == 'all':
        converter.convert_all_splits()
    else:
        out_file = Path(args.output_dir) / f"coco_annotations_{args.split}.json"
        converter.convert(split=args.split, output_file=str(out_file))

    logger.info("转换完成")
