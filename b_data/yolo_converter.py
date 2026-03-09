"""
YOLO 格式转换器（生产级）

功能：
1. 将 COCO 标注转换为 YOLO 检测 / 分割格式
2. 安全的目录操作（不再 rmtree）
3. 结构化日志，统计信息作为返回值
4. 支持 symlink 替代复制以减少磁盘 I/O

使用方法：
    from data.yolo_converter import YOLOConverter, YOLOConverterConfig

    converter = YOLOConverter(data_root)
    result, stats = converter.convert_from_coco('coco_train.json', output_dir='yolo/train')
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from .schemas import ConversionStats
from .exceptions import ConversionError

logger = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class YOLOConverterConfig:
    """YOLO 转换器配置"""
    # 输出格式
    use_segmentation: bool = True

    # 图像处理
    copy_images: bool = True
    rename_images: bool = True
    use_symlink: bool = False  # True = 软链接替代复制，节省磁盘

    # 验证
    validate_coords: bool = True
    coord_tolerance: float = 1.1  # 坐标超出图像尺寸的容差倍数
    min_polygon_area: float = 1e-6

    # 输出
    output_dir: str = "./yolo_dataset"

    # 安全选项
    allow_overwrite: bool = False  # 若 True 才允许清空已有输出目录


# ============================================================================
# YOLO 转换器
# ============================================================================

class YOLOConverter:
    """将 COCO 格式标注转换为 YOLO 格式。"""

    def __init__(self, data_root: str, config: Optional[YOLOConverterConfig] = None):
        self.data_root = Path(data_root)
        self.config = config or YOLOConverterConfig()

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def convert_from_coco(
        self,
        coco_json: str,
        output_dir: Optional[str] = None,
        split_name: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], ConversionStats]:
        """
        从 COCO JSON 转换为 YOLO 格式。

        Returns:
            (result_info, stats)
        """
        output_dir_path = Path(output_dir or self.config.output_dir)
        mode_str = '分割' if self.config.use_segmentation else '检测'
        logger.info("COCO → YOLO 转换 | 输入: %s | 输出: %s | 格式: %s",
                     coco_json, output_dir_path, mode_str)

        # 加载 COCO
        coco_path = Path(coco_json)
        if not coco_path.exists():
            raise ConversionError(f"COCO JSON 不存在: {coco_path}")

        with open(coco_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # 准备输出目录（安全模式）
        images_dir, labels_dir = self._prepare_output_dirs(output_dir_path)

        # 类别映射
        sorted_cats = sorted(coco_data['categories'], key=lambda x: x['id'])
        cat_id_to_yolo = {cat['id']: idx for idx, cat in enumerate(sorted_cats)}

        logger.info("类别映射 (YOLO → COCO → Name):")
        for cat in sorted_cats:
            logger.info("  %d → %d → %s", cat_id_to_yolo[cat['id']], cat['id'], cat['name'])

        # 按图像分组标注
        img_to_anns: Dict[int, List[Dict]] = {}
        for ann in coco_data['annotations']:
            img_to_anns.setdefault(ann['image_id'], []).append(ann)

        # 处理
        stats = ConversionStats()

        for img_info in tqdm(coco_data['images'], desc="YOLO 转换"):
            img_stats = self._process_image(
                img_info=img_info,
                annotations=img_to_anns.get(img_info['id'], []),
                cat_id_to_yolo=cat_id_to_yolo,
                images_dir=images_dir,
                labels_dir=labels_dir,
            )
            stats = stats.merge(img_stats)

        logger.info("YOLO 转换统计:\n%s", stats.summary())

        result_info = {
            'images_dir': str(images_dir),
            'labels_dir': str(labels_dir),
            'num_classes': len(sorted_cats),
            'class_names': [cat['name'] for cat in sorted_cats],
        }
        return result_info, stats

    def convert_all_splits(
        self,
        coco_dir: str,
        output_dir: Optional[str] = None,
        splits: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """转换所有划分并生成 YAML 配置。"""
        splits = splits or ['train', 'val', 'test']
        coco_dir_path = Path(coco_dir)
        out = Path(output_dir or self.config.output_dir)

        results = {}
        for split in splits:
            coco_json = coco_dir_path / f"coco_annotations_{split}.json"
            if not coco_json.exists():
                logger.warning("跳过: %s 不存在", coco_json)
                continue
            info, stats = self.convert_from_coco(
                coco_json=str(coco_json),
                output_dir=str(out / split),
                split_name=split,
            )
            results[split] = info

        if results:
            self._create_yaml_config(out, results)

        logger.info("所有划分已转换完成，输出目录: %s", out)
        return results

    # ------------------------------------------------------------------
    # 图像处理
    # ------------------------------------------------------------------

    def _process_image(
        self,
        img_info: Dict,
        annotations: List[Dict],
        cat_id_to_yolo: Dict[int, int],
        images_dir: Path,
        labels_dir: Path,
    ) -> ConversionStats:
        """处理单张图像，返回本张图的统计。"""
        stats = ConversionStats()
        file_name = img_info['file_name'].lstrip('/')
        img_w = int(img_info['width'])
        img_h = int(img_info['height'])
        img_id = img_info['id']

        # 查找源图像
        src_path = self._find_source_image(file_name)
        if src_path is None:
            stats.skipped_samples += 1
            return stats

        # 复制 / 链接图像
        dst_img_name = f"{img_id:06d}{src_path.suffix}" if self.config.rename_images else src_path.name
        if self.config.copy_images:
            self._copy_or_link(src_path, images_dir / dst_img_name)

        # 标签文件
        label_name = f"{img_id:06d}.txt" if self.config.rename_images else f"{src_path.stem}.txt"
        label_path = labels_dir / label_name

        if not annotations:
            label_path.write_text("")
            stats.total_images += 1
            return stats

        lines: List[str] = []
        for ann in annotations:
            cat_id = ann.get('category_id')
            if cat_id not in cat_id_to_yolo:
                stats.invalid_polygons += 1
                continue

            class_idx = cat_id_to_yolo[cat_id]
            stats.total_annotations += 1

            if self.config.use_segmentation:
                line = self._seg_line(ann, class_idx, img_w, img_h, stats)
            else:
                line = self._det_line(ann, class_idx, img_w, img_h, stats)

            if line:
                lines.append(line)

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        stats.total_images += 1
        return stats

    # ------------------------------------------------------------------
    # 标签行生成
    # ------------------------------------------------------------------

    def _seg_line(
        self, ann: Dict, class_idx: int, img_w: int, img_h: int,
        stats: ConversionStats,
    ) -> Optional[str]:
        """生成分割格式标签行: class_idx x1 y1 x2 y2 ..."""
        polygon_coords = self._extract_seg_coords(ann, img_w, img_h, stats)

        if polygon_coords is None:
            # 回退: bbox → 矩形 polygon
            bbox = ann.get('bbox')
            if bbox and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                polygon_coords = self._bbox_to_polygon(bbox, img_w, img_h)
                if polygon_coords:
                    stats.annotations_with_seg += 1  # bbox_to_poly

        if polygon_coords and len(polygon_coords) >= 6:
            coords_str = ' '.join(f"{c:.6f}" for c in polygon_coords)
            return f"{class_idx} {coords_str}"

        stats.invalid_polygons += 1
        return None

    def _det_line(
        self, ann: Dict, class_idx: int, img_w: int, img_h: int,
        stats: ConversionStats,
    ) -> Optional[str]:
        """生成检测格式标签行: class_idx x_center y_center w h"""
        bbox = ann.get('bbox')
        if not bbox or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            stats.invalid_polygons += 1
            return None

        xc = np.clip((bbox[0] + bbox[2] / 2) / img_w, 0.0, 1.0)
        yc = np.clip((bbox[1] + bbox[3] / 2) / img_h, 0.0, 1.0)
        w = np.clip(bbox[2] / img_w, 0.0, 1.0)
        h = np.clip(bbox[3] / img_h, 0.0, 1.0)

        return f"{class_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

    def _extract_seg_coords(
        self, ann: Dict, img_w: int, img_h: int, stats: ConversionStats
    ) -> Optional[List[float]]:
        """从 COCO segmentation 提取归一化坐标。"""
        seg_data = ann.get('segmentation')
        if not seg_data or not isinstance(seg_data, list) or not seg_data:
            return None

        for seg in seg_data:
            if not isinstance(seg, list) or len(seg) < 6:
                continue
            if len(seg) % 2 != 0:
                seg = seg[:-1]
            if len(seg) < 6:
                continue

            try:
                points = np.array(seg, dtype=np.float64).reshape(-1, 2)

                if self.config.validate_coords:
                    tol = self.config.coord_tolerance
                    if points[:, 0].max() > img_w * tol or points[:, 1].max() > img_h * tol:
                        stats.invalid_polygons += 1
                        continue

                points[:, 0] = np.clip(points[:, 0], 0, img_w) / float(img_w)
                points[:, 1] = np.clip(points[:, 1], 0, img_h) / float(img_h)
                points = np.clip(points, 0.0, 1.0)

                if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                    stats.invalid_polygons += 1
                    continue

                x, y = points[:, 0], points[:, 1]
                area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                if area < self.config.min_polygon_area:
                    stats.invalid_polygons += 1
                    continue

                stats.annotations_with_seg += 1
                return points.flatten().tolist()

            except (ValueError, TypeError):
                stats.invalid_polygons += 1

        return None

    @staticmethod
    def _bbox_to_polygon(bbox: List[float], img_w: int, img_h: int) -> Optional[List[float]]:
        """将 COCO bbox [x,y,w,h] 转为归一化矩形 polygon。"""
        x_min, y_min, w, h = bbox
        x_max, y_max = x_min + w, y_min + h

        x_min = max(0.0, min(x_min, img_w))
        y_min = max(0.0, min(y_min, img_h))
        x_max = max(0.0, min(x_max, img_w))
        y_max = max(0.0, min(y_max, img_h))

        if x_max <= x_min or y_max <= y_min:
            return None

        pts = np.array([
            [x_min, y_min], [x_max, y_min],
            [x_max, y_max], [x_min, y_max],
        ], dtype=np.float64)

        pts[:, 0] /= img_w
        pts[:, 1] /= img_h
        return np.clip(pts, 0.0, 1.0).flatten().tolist()

    # ------------------------------------------------------------------
    # 目录与文件操作
    # ------------------------------------------------------------------

    def _prepare_output_dirs(self, output_dir: Path) -> Tuple[Path, Path]:
        """
        安全地准备输出目录。

        不再使用 rmtree —— 仅当 allow_overwrite=True 时才清空。
        """
        images_dir = output_dir / 'images'
        labels_dir = output_dir / 'labels'

        if output_dir.exists():
            if self.config.allow_overwrite:
                logger.warning("清空已有输出目录: %s", output_dir)
                shutil.rmtree(output_dir)
            else:
                logger.info("输出目录已存在，将追加/覆盖文件: %s", output_dir)

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        return images_dir, labels_dir

    def _find_source_image(self, file_name: str) -> Optional[Path]:
        """查找源图像文件。"""
        candidates = [
            self.data_root / file_name,
            self.data_root / file_name.replace('/', os.sep),
        ]
        for p in candidates:
            if p.exists():
                return p
        logger.debug("找不到源图像: %s", file_name)
        return None

    def _copy_or_link(self, src: Path, dst: Path) -> None:
        """复制或软链接图像文件。"""
        if dst.exists():
            return
        if self.config.use_symlink:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)

    # ------------------------------------------------------------------
    # YAML 配置
    # ------------------------------------------------------------------

    def _create_yaml_config(self, output_dir: Path, results: Dict[str, Any]) -> None:
        first = next(iter(results.values()))
        names = first['class_names']

        lines = [
            "# CubiCasa5K Floorplan Dataset (auto-generated)",
            f"path: {output_dir.absolute()}",
            "train: train/images",
            "val: val/images",
            "test: test/images",
            f"nc: {first['num_classes']}",
            "names:",
        ]
        for i, name in enumerate(names):
            lines.append(f"  {i}: {name}")

        yaml_file = output_dir / 'dataset.yaml'
        yaml_file.write_text("\n".join(lines) + "\n", encoding='utf-8')
        logger.info("YAML 配置已创建: %s", yaml_file)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description='COCO → YOLO 格式转换')
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--coco_dir', type=str, required=True)
    ap.add_argument('--output_dir', type=str, default='./yolo_dataset')
    ap.add_argument('--no_seg', action='store_true', help='使用检测格式')
    ap.add_argument('--no_copy', action='store_true', help='不复制图像')
    ap.add_argument('--symlink', action='store_true', help='使用软链接替代复制')
    ap.add_argument('--overwrite', action='store_true', help='允许清空已有输出目录')
    args = ap.parse_args()

    cfg = YOLOConverterConfig(
        use_segmentation=not args.no_seg,
        copy_images=not args.no_copy,
        use_symlink=args.symlink,
        output_dir=args.output_dir,
        allow_overwrite=args.overwrite,
    )
    converter = YOLOConverter(args.data_root, cfg)
    converter.convert_all_splits(coco_dir=args.coco_dir, output_dir=args.output_dir)
    logger.info("转换完成")
