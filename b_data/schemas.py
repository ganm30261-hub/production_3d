"""
数据结构定义（Schema）

使用 dataclass 和类型注解定义所有中间数据结构，
替代原先的 Dict[str, Any]，确保类型安全和数据验证。
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================================
# 基础校验工具
# ============================================================================

def validate_polygon(polygon: List[List[float]], min_points: int = 3) -> List[List[float]]:
    """
    校验多边形坐标的有效性。

    Args:
        polygon: 多边形顶点列表，每个元素为 [x, y]
        min_points: 最少顶点数

    Returns:
        校验通过的原始 polygon

    Raises:
        ValueError: 坐标无效时抛出
    """
    if len(polygon) < min_points:
        raise ValueError(
            f"polygon 至少需要 {min_points} 个顶点，实际 {len(polygon)}"
        )
    for i, point in enumerate(polygon):
        if len(point) != 2:
            raise ValueError(f"第 {i} 个顶点维度应为 2，实际 {len(point)}")
        if any(math.isnan(v) or math.isinf(v) for v in point):
            raise ValueError(f"第 {i} 个顶点包含 NaN 或 Inf: {point}")
    return polygon


def validate_bbox(bbox: List[float]) -> List[float]:
    """
    校验边界框格式：[x_min, y_min, x_max, y_max]

    Raises:
        ValueError: bbox 无效时抛出
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox 长度应为 4，实际 {len(bbox)}")
    x_min, y_min, x_max, y_max = bbox
    if any(math.isnan(v) or math.isinf(v) for v in bbox):
        raise ValueError(f"bbox 包含 NaN 或 Inf: {bbox}")
    if x_max < x_min or y_max < y_min:
        raise ValueError(f"bbox 坐标顺序错误: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
    return bbox


# ============================================================================
# 标注数据结构
# ============================================================================

@dataclass
class RoomAnnotation:
    """单个房间的标注数据"""
    label_id: int
    room_type: str
    polygon: List[List[float]]
    polygon_area: float
    num_vertices: int
    bbox: List[float]
    bbox_area: float
    center: List[float]

    def __post_init__(self):
        validate_polygon(self.polygon, min_points=3)
        validate_bbox(self.bbox)


@dataclass
class WallAnnotation:
    """单段墙体的标注数据"""
    wall_type: str = "Wall"
    polygon: List[List[float]] = field(default_factory=list)
    thickness: float = 0.0
    length: float = 0.0
    height: float = 2.8

    def __post_init__(self):
        if self.polygon:
            validate_polygon(self.polygon, min_points=2)


@dataclass
class OpeningAnnotation:
    """门或窗的标注数据"""
    opening_type: str  # "Door" or "Window"
    bbox: List[float] = field(default_factory=list)
    center: List[float] = field(default_factory=list)
    area: float = 0.0
    width: float = 0.0
    height: float = 0.0
    sill_height: float = 0.0  # 仅窗户有效

    def __post_init__(self):
        if self.bbox:
            validate_bbox(self.bbox)


@dataclass
class StructuralAnnotation:
    """结构元素标注（栏杆、柱子）"""
    element_type: str  # "Railing" or "Column"
    polygon: List[List[float]] = field(default_factory=list)
    height: float = 0.0
    thickness: float = 0.0


@dataclass
class SampleData:
    """
    单个样本的完整数据，替代原先的 Dict[str, Any]。

    这是 CubiCasaParser.load_sample() 的返回类型。
    """
    image: np.ndarray
    image_path: str
    sample_path: str
    height: int
    width: int
    coef_width: float = 1.0
    coef_height: float = 1.0
    rooms: List[RoomAnnotation] = field(default_factory=list)
    walls: List[WallAnnotation] = field(default_factory=list)
    doors: List[OpeningAnnotation] = field(default_factory=list)
    windows: List[OpeningAnnotation] = field(default_factory=list)
    structural_railings: List[StructuralAnnotation] = field(default_factory=list)
    structural_columns: List[StructuralAnnotation] = field(default_factory=list)
    house_walls: Optional[np.ndarray] = None
    house_icons: Optional[np.ndarray] = None

    # 论文方法：墙体二值掩码 (H,W) uint8, 0/255
    # 由 (house.walls == 2) * 255 生成，供 FPN 语义分割训练
    wall_mask: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """向后兼容：转换为旧版字典格式"""
        from dataclasses import asdict
        d = {
            'image': self.image,
            'image_path': self.image_path,
            'sample_path': self.sample_path,
            'height': self.height,
            'width': self.width,
            'coef_width': self.coef_width,
            'coef_height': self.coef_height,
            'rooms': [vars(r) for r in self.rooms],
            'walls': [vars(w) for w in self.walls],
            'doors': [vars(d) for d in self.doors],
            'windows': [vars(w) for w in self.windows],
            'structural': {
                'railings': [vars(r) for r in self.structural_railings],
                'columns': [vars(c) for c in self.structural_columns],
            },
            'house_walls': self.house_walls,
            'house_icons': self.house_icons,
            'wall_mask': self.wall_mask,
        }
        return d


# ============================================================================
# 转换统计
# ============================================================================

@dataclass
class ConversionStats:
    """格式转换的统计信息（不可变返回值，不再用实例变量累积）"""
    total_images: int = 0
    total_annotations: int = 0
    annotations_with_seg: int = 0
    annotations_without_seg: int = 0
    invalid_polygons: int = 0
    skipped_samples: int = 0
    error_samples: int = 0
    category_counts: Dict[str, int] = field(default_factory=dict)

    def merge(self, other: "ConversionStats") -> "ConversionStats":
        """合并两个统计对象（用于并行处理后汇总）"""
        merged_cats = dict(self.category_counts)
        for k, v in other.category_counts.items():
            merged_cats[k] = merged_cats.get(k, 0) + v
        return ConversionStats(
            total_images=self.total_images + other.total_images,
            total_annotations=self.total_annotations + other.total_annotations,
            annotations_with_seg=self.annotations_with_seg + other.annotations_with_seg,
            annotations_without_seg=self.annotations_without_seg + other.annotations_without_seg,
            invalid_polygons=self.invalid_polygons + other.invalid_polygons,
            skipped_samples=self.skipped_samples + other.skipped_samples,
            error_samples=self.error_samples + other.error_samples,
            category_counts=merged_cats,
        )

    def summary(self) -> str:
        total = max(self.total_annotations, 1)
        lines = [
            f"  图像: {self.total_images}",
            f"  标注: {self.total_annotations}",
            f"  有分割: {self.annotations_with_seg} ({self.annotations_with_seg / total * 100:.1f}%)",
            f"  无分割: {self.annotations_without_seg} ({self.annotations_without_seg / total * 100:.1f}%)",
        ]
        if self.invalid_polygons > 0:
            lines.append(f"  无效 polygon: {self.invalid_polygons}")
        if self.skipped_samples > 0:
            lines.append(f"  跳过样本: {self.skipped_samples}")
        if self.error_samples > 0:
            lines.append(f"  错误样本: {self.error_samples}")
        if self.category_counts:
            lines.append("  类别分布:")
            for cat, count in sorted(self.category_counts.items()):
                lines.append(f"    {cat:12s}: {count:5d}")
        return "\n".join(lines)
