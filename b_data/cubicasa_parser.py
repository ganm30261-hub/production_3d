"""
CubiCasa5k SVG 解析器（生产级）

功能：
1. 解析 SVG 标注文件，提取房间、墙体、门窗信息
2. 支持矢量化提取（直接从 SVG）和掩码提取（从像素）
3. 与 floortrans.loaders.house 可选集成
4. 带结构化日志、自定义异常、Schema 校验、LRU 缓存

使用方法：
    from data.cubicasa_parser import CubiCasaParser, ParserConfig
    
    parser = CubiCasaParser(data_root)
    sample = parser.load_sample("high_quality_architectural/6044")
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from xml.dom import minidom
from xml.parsers.expat import ExpatError

import cv2
import numpy as np

from .exceptions import (
    SVGParseError,
    ImageLoadError,
    SplitFileError,
    InvalidAnnotationError,
)
from .schemas import (
    RoomAnnotation,
    WallAnnotation,
    OpeningAnnotation,
    StructuralAnnotation,
    SampleData,
)

logger = logging.getLogger(__name__)

# PyTorch（可选）
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# 配置
# ============================================================================

@dataclass
class ParserConfig:
    """解析器配置——所有可调参数集中管理"""

    # 文件名约定
    scaled_image: str = "F1_scaled.png"
    original_image: str = "F1_original.png"
    svg_file: str = "model.svg"

    # 划分文件
    split_files: Dict[str, str] = field(default_factory=lambda: {
        'train': 'train_hq_arch.txt',
        'val': 'val_hq_arch.txt',
        'test': 'test_hq_arch.txt',
    })

    # 提取开关
    extract_rooms: bool = True
    extract_walls: bool = True
    extract_doors_windows: bool = True
    extract_structural: bool = True
    use_original_size: bool = False

    # 阈值——从代码中收归配置
    room_contour_min_area: float = 100.0
    wall_contour_min_area: float = 50.0
    opening_min_area: float = 10.0
    contour_approx_epsilon_ratio: float = 0.01
    default_wall_height: float = 2.8
    default_door_width: float = 0.9
    default_door_height: float = 2.1
    default_window_height: float = 1.2
    default_sill_height: float = 0.9
    default_railing_height: float = 1.1
    default_railing_thickness: float = 0.05

    # 缓存
    enable_cache: bool = True
    cache_maxsize: int = 256


# 房间类型映射
ROOM_TYPES: Dict[int, str] = {
    3: "Kitchen",
    4: "LivingRoom",
    5: "Bedroom",
    6: "Bath",
    7: "Entry",
    9: "Storage",
    10: "Garage",
}

ROOM_NAME_TO_ID: Dict[str, int] = {v: k for k, v in ROOM_TYPES.items()}

# 掩码标签值
DOOR_LABEL = 2
WINDOW_LABEL = 1
WALL_LABEL = 2
RAILING_LABEL = 8


# ============================================================================
# SVG 解析器
# ============================================================================

class SVGParser:
    """
    SVG 文件矢量解析器。

    直接解析 model.svg 提取多边形信息，避免像素化失真。
    """

    def __init__(self, svg_path: str, img_height: int, img_width: int):
        self.svg_path = svg_path
        self.img_height = img_height
        self.img_width = img_width
        try:
            self.svg = minidom.parse(svg_path)
        except ExpatError as e:
            raise SVGParseError(f"SVG XML 解析失败 [{svg_path}]: {e}") from e
        except FileNotFoundError as e:
            raise SVGParseError(f"SVG 文件不存在 [{svg_path}]") from e

    def extract_rooms(
        self,
        coef_width: float = 1.0,
        coef_height: float = 1.0,
    ) -> List[RoomAnnotation]:
        """
        从 SVG 提取房间标注。

        Args:
            coef_width: 宽度缩放系数
            coef_height: 高度缩放系数

        Returns:
            RoomAnnotation 列表
        """
        rooms: List[RoomAnnotation] = []

        for space_elem in self.svg.getElementsByTagName('g'):
            if 'Space' not in space_elem.getAttribute('class'):
                continue

            class_attr = space_elem.getAttribute('class')
            room_type = class_attr.replace('Space ', '').split()[0]

            polygons = space_elem.getElementsByTagName('polygon')
            if not polygons:
                continue

            points_str = polygons[0].getAttribute('points').strip()
            if not points_str:
                continue

            polygon_coords = self._parse_polygon_points(points_str, coef_width, coef_height)
            if len(polygon_coords) < 3:
                logger.debug("跳过不足 3 个顶点的房间 polygon: %s", room_type)
                continue

            polygon_array = np.array(polygon_coords)
            x_coords = polygon_array[:, 0]
            y_coords = polygon_array[:, 1]

            bbox = [
                float(x_coords.min()),
                float(y_coords.min()),
                float(x_coords.max()),
                float(y_coords.max()),
            ]
            area = self._compute_polygon_area(x_coords, y_coords)

            try:
                room = RoomAnnotation(
                    label_id=ROOM_NAME_TO_ID.get(room_type, 11),
                    room_type=room_type,
                    polygon=polygon_coords,
                    polygon_area=float(area),
                    num_vertices=len(polygon_coords),
                    bbox=bbox,
                    bbox_area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    center=[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                )
                rooms.append(room)
            except (ValueError, InvalidAnnotationError) as e:
                logger.warning("房间标注校验失败 [%s]: %s", room_type, e)
                continue

        return rooms

    def extract_walls(
        self,
        coef_width: float = 1.0,
        coef_height: float = 1.0,
        default_height: float = 2.8,
    ) -> List[WallAnnotation]:
        """从 SVG 提取墙体标注。"""
        walls: List[WallAnnotation] = []

        for wall_elem in self.svg.getElementsByTagName('g'):
            class_attr = wall_elem.getAttribute('class')
            if 'Wall' not in class_attr:
                continue

            polygons = wall_elem.getElementsByTagName('polygon')
            if not polygons:
                continue

            points_str = polygons[0].getAttribute('points').strip()
            if not points_str:
                continue

            polygon_coords = self._parse_polygon_points(points_str, coef_width, coef_height)
            if len(polygon_coords) < 2:
                continue

            polygon_array = np.array(polygon_coords)

            if len(polygon_coords) >= 4:
                w = np.max(polygon_array[:, 0]) - np.min(polygon_array[:, 0])
                h = np.max(polygon_array[:, 1]) - np.min(polygon_array[:, 1])
                thickness = float(min(w, h))
                length = float(max(w, h))
            else:
                thickness = 0.0
                length = float(np.linalg.norm(polygon_array[-1] - polygon_array[0]))

            walls.append(WallAnnotation(
                wall_type='Wall',
                polygon=polygon_coords,
                thickness=thickness,
                length=length,
                height=default_height,
            ))

        return walls

    def extract_doors_windows(
        self,
        coef_width: float = 1.0,
        coef_height: float = 1.0,
    ) -> Tuple[List[OpeningAnnotation], List[OpeningAnnotation]]:
        """
        从 SVG 提取门窗标注（降级方案，House 类不可用时使用）。

        解析 SVG 中 class 包含 'Door' 或 'Window' 的 <g> 元素，
        从其子 polygon 计算边界框。

        Returns:
            (doors, windows) 元组
        """
        doors: List[OpeningAnnotation] = []
        windows: List[OpeningAnnotation] = []

        for elem in self.svg.getElementsByTagName('g'):
            class_attr = elem.getAttribute('class')
            is_door = 'Door' in class_attr
            is_window = 'Window' in class_attr
            if not is_door and not is_window:
                continue

            polygons = elem.getElementsByTagName('polygon')
            if not polygons:
                continue

            points_str = polygons[0].getAttribute('points').strip()
            if not points_str:
                continue

            coords = self._parse_polygon_points(points_str, coef_width, coef_height)
            if len(coords) < 2:
                continue

            pa = np.array(coords)
            x_min, y_min = float(pa[:, 0].min()), float(pa[:, 1].min())
            x_max, y_max = float(pa[:, 0].max()), float(pa[:, 1].max())
            bw, bh = x_max - x_min, y_max - y_min
            if bw <= 0 or bh <= 0:
                continue

            opening = OpeningAnnotation(
                opening_type='Door' if is_door else 'Window',
                bbox=[x_min, y_min, x_max, y_max],
                center=[(x_min + x_max) / 2, (y_min + y_max) / 2],
                area=float(bw * bh),
                width=float(max(bw, bh)),
                height=float(min(bw, bh)),
            )

            if is_door:
                doors.append(opening)
            else:
                windows.append(opening)

        return doors, windows

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_polygon_points(
        points_str: str,
        coef_width: float,
        coef_height: float,
    ) -> List[List[float]]:
        """解析 SVG polygon 的 points 属性。"""
        coords: List[List[float]] = []
        for pair in points_str.split():
            if ',' in pair:
                x_str, y_str = pair.split(',')
                coords.append([float(x_str) * coef_width, float(y_str) * coef_height])
        return coords

    @staticmethod
    def _compute_polygon_area(x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """Shoelace 公式计算多边形面积。"""
        n = len(x_coords)
        area = 0.5 * abs(sum(
            x_coords[i] * y_coords[(i + 1) % n] - x_coords[(i + 1) % n] * y_coords[i]
            for i in range(n)
        ))
        return area


# ============================================================================
# CubiCasa 解析器
# ============================================================================

class CubiCasaParser:
    """
    CubiCasa5k 数据集主解析器。

    整合 House 类（可选）和 SVG 直接解析，提供完整的样本加载功能。
    """

    def __init__(self, data_root: str, config: Optional[ParserConfig] = None):
        self.data_root = Path(data_root)
        self.config = config or ParserConfig()

        if not self.data_root.exists():
            raise FileNotFoundError(f"数据集根目录不存在: {self.data_root}")

        # 尝试导入 House 类
        self.house_available = False
        import sys
        sys.path.insert(0, '/CubiCasa5k')
        try:
            from floortrans.loaders.house import House
            self.House = House
            self.house_available = True
            logger.info("floortrans.loaders.house 可用")
        except ImportError:
            logger.info("floortrans.loaders.house 不可用，将使用纯 SVG 解析")

        # 划分文件缓存
        self._split_cache: Dict[str, List[str]] = {}

    def load_split_file(self, split: str = 'train') -> List[str]:
        """
        加载划分文件，返回样本路径列表。

        Args:
            split: 数据划分名称

        Returns:
            样本相对路径列表

        Raises:
            SplitFileError: 划分文件不存在或为空
        """
        if split in self._split_cache:
            return self._split_cache[split]

        filename = self.config.split_files.get(split, f'{split}.txt')
        split_file = self.data_root / filename

        if not split_file.exists():
            raise SplitFileError(f"找不到划分文件: {split_file}")

        samples: List[str] = []
        with open(split_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    samples.append(line.lstrip('/').rstrip('/'))

        if not samples:
            raise SplitFileError(f"划分文件为空: {split_file}")

        logger.info("加载 %s 集: %d 个样本 (from %s)", split, len(samples), split_file)
        self._split_cache[split] = samples
        return samples

    def load_sample(self, sample_path: str) -> SampleData:
        """
        加载单个样本的完整数据。

        Args:
            sample_path: 样本相对路径，如 "high_quality_architectural/6044"

        Returns:
            SampleData 对象

        Raises:
            ImageLoadError: 图像无法加载
            SVGParseError: SVG 解析失败
        """
        sample_dir = self.data_root / sample_path

        # ── 加载图像 ──
        image, image_path = self._load_image(sample_dir)
        height, width = image.shape[:2]

        # ── 计算缩放系数 ──
        coef_width, coef_height = self._compute_scale_coefs(sample_dir, height, width)

        # ── 解析标注 ──
        svg_path = sample_dir / self.config.svg_file

        result = SampleData(
            image=image,
            image_path=str(image_path),
            sample_path=sample_path,
            height=height,
            width=width,
            coef_width=coef_width,
            coef_height=coef_height,
        )

        if self.house_available and svg_path.exists():
            self._parse_with_house(result, sample_dir, svg_path, height, width, coef_width, coef_height)
        else:
            self._parse_svg_only(result, svg_path, height, width, coef_width, coef_height)

        return result

    # ------------------------------------------------------------------
    # 图像加载
    # ------------------------------------------------------------------

    def _load_image(self, sample_dir: Path) -> Tuple[np.ndarray, Path]:
        """按优先级尝试加载图像文件。"""
        if self.config.use_original_size:
            candidates = [self.config.original_image, self.config.scaled_image]
        else:
            candidates = [self.config.scaled_image, self.config.original_image]

        for fname in candidates:
            path = sample_dir / fname
            if path.exists():
                image = cv2.imread(str(path))
                if image is not None:
                    return image, path
                raise ImageLoadError(f"cv2.imread 返回 None: {path}")

        raise ImageLoadError(
            f"找不到任何可用图像文件: {sample_dir} (尝试了 {candidates})"
        )

    def _compute_scale_coefs(
        self, sample_dir: Path, height: int, width: int
    ) -> Tuple[float, float]:
        """计算原始图像与 scaled 图像之间的缩放系数。"""
        if not self.config.use_original_size:
            return 1.0, 1.0

        scaled_path = sample_dir / self.config.scaled_image
        if not scaled_path.exists():
            return 1.0, 1.0

        scaled_img = cv2.imread(str(scaled_path))
        if scaled_img is None:
            return 1.0, 1.0

        scaled_h, scaled_w = scaled_img.shape[:2]
        if scaled_w == 0 or scaled_h == 0:
            return 1.0, 1.0

        return width / scaled_w, height / scaled_h

    # ------------------------------------------------------------------
    # House 类解析路径
    # ------------------------------------------------------------------

    def _parse_with_house(
        self,
        result: SampleData,
        sample_dir: Path,
        svg_path: Path,
        height: int,
        width: int,
        coef_width: float,
        coef_height: float,
    ) -> None:
        """使用 floortrans House 类解析（精度更高）。"""
        try:
            if self.config.use_original_size:
                scaled_path = sample_dir / self.config.scaled_image
                if scaled_path.exists():
                    scaled_img = cv2.imread(str(scaled_path))
                    if scaled_img is not None:
                        h, w = scaled_img.shape[:2]
                    else:
                        h, w = height, width
                else:
                    h, w = height, width
            else:
                h, w = height, width

            house = self.House(str(svg_path), h, w)

            result.house_walls = house.walls
            result.house_icons = house.icons

            # 论文方法：生成墙体二值掩码 (FPN 语义分割训练标签)
            # house.walls 是多类别掩码, label==2 是墙壁
            wall_binary = (house.walls == WALL_LABEL).astype(np.uint8) * 255
            if self.config.use_original_size and (coef_width != 1.0 or coef_height != 1.0):
                wall_binary = cv2.resize(
                    wall_binary, (width, height), interpolation=cv2.INTER_NEAREST,
                )
            result.wall_mask = wall_binary

            if self.config.extract_rooms:
                result.rooms = self._extract_rooms(house, svg_path, coef_width, coef_height)

            if self.config.extract_walls:
                result.walls = self._extract_walls_from_mask(house.walls, coef_width, coef_height)

            if self.config.extract_doors_windows:
                result.doors, result.windows = self._extract_doors_windows(
                    house, coef_width, coef_height
                )

            if self.config.extract_structural:
                result.structural_railings = self._extract_railings(
                    house, coef_width, coef_height
                )

        except SVGParseError:
            raise
        except Exception as e:
            logger.warning(
                "House 解析失败 [%s]，回退到 SVG 解析: %s",
                svg_path, e, exc_info=True,
            )
            self._parse_svg_only(result, svg_path, height, width, coef_width, coef_height)

    # ------------------------------------------------------------------
    # 纯 SVG 解析路径（降级方案）
    # ------------------------------------------------------------------

    def _parse_svg_only(
        self,
        result: SampleData,
        svg_path: Path,
        height: int,
        width: int,
        coef_width: float,
        coef_height: float,
    ) -> None:
        """不依赖 House 类的降级解析。"""
        if not svg_path.exists():
            logger.warning("SVG 文件不存在，跳过标注提取: %s", svg_path)
            return

        try:
            svg_parser = SVGParser(str(svg_path), height, width)
            if self.config.extract_rooms:
                result.rooms = svg_parser.extract_rooms(coef_width, coef_height)
            if self.config.extract_walls:
                result.walls = svg_parser.extract_walls(
                    coef_width, coef_height,
                    default_height=self.config.default_wall_height,
                )
                # 从墙体多边形渲染二值掩码（降级方案）
                mask = np.zeros((height, width), dtype=np.uint8)
                for wall in result.walls:
                    if len(wall.polygon) >= 3:
                        pts = np.array(wall.polygon, dtype=np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                result.wall_mask = mask
            if self.config.extract_doors_windows:
                result.doors, result.windows = svg_parser.extract_doors_windows(
                    coef_width, coef_height,
                )
        except SVGParseError as e:
            logger.error("SVG 解析失败: %s", e)

    # ------------------------------------------------------------------
    # 各类标注提取方法
    # ------------------------------------------------------------------

    def _extract_rooms(
        self, house, svg_path: Path, coef_width: float, coef_height: float
    ) -> List[RoomAnnotation]:
        """优先 SVG 矢量提取，失败回退到掩码提取。"""
        try:
            svg_parser = SVGParser(str(svg_path), house.walls.shape[0], house.walls.shape[1])
            return svg_parser.extract_rooms(coef_width, coef_height)
        except SVGParseError as e:
            logger.debug("SVG 房间提取失败，回退到掩码: %s", e)
            return self._extract_rooms_from_mask(house.walls, coef_width, coef_height)

    def _extract_rooms_from_mask(
        self, walls_mask: np.ndarray, coef_width: float, coef_height: float
    ) -> List[RoomAnnotation]:
        """从语义掩码提取房间轮廓。"""
        rooms: List[RoomAnnotation] = []
        cfg = self.config

        for label_id in range(3, 12):
            mask = (walls_mask == label_id).astype(np.uint8)
            if not mask.any():
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < cfg.room_contour_min_area:
                    continue

                epsilon = cfg.contour_approx_epsilon_ratio * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                polygon = approx.squeeze().astype(float)

                if polygon.ndim == 1:
                    continue

                polygon[:, 0] *= coef_width
                polygon[:, 1] *= coef_height

                x, y, w, h = cv2.boundingRect(contour)
                bbox = [
                    x * coef_width, y * coef_height,
                    (x + w) * coef_width, (y + h) * coef_height,
                ]

                try:
                    room = RoomAnnotation(
                        label_id=label_id,
                        room_type=ROOM_TYPES.get(label_id, 'Room'),
                        polygon=polygon.tolist(),
                        polygon_area=float(area) * coef_width * coef_height,
                        num_vertices=len(polygon),
                        bbox=bbox,
                        bbox_area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                        center=[(x + w / 2) * coef_width, (y + h / 2) * coef_height],
                    )
                    rooms.append(room)
                except ValueError as e:
                    logger.debug("掩码房间校验失败 label=%d: %s", label_id, e)

        return rooms

    def _extract_walls_from_mask(
        self, walls_mask, coef_width: float, coef_height: float
    ) -> List[WallAnnotation]:
        """从 house.walls 掩码提取墙体轮廓。"""
        cfg = self.config
        walls: List[WallAnnotation] = []

        wall_mask = (walls_mask == WALL_LABEL).astype(np.uint8)
        contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < cfg.wall_contour_min_area:
                continue

            epsilon = cfg.contour_approx_epsilon_ratio * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygon = approx.squeeze().astype(float)

            if polygon.ndim == 1:
                continue

            polygon[:, 0] *= coef_width
            polygon[:, 1] *= coef_height

            rect = cv2.minAreaRect(contour)
            width_rect = float(min(rect[1]) * coef_width)
            length_rect = float(max(rect[1]) * coef_height)

            walls.append(WallAnnotation(
                wall_type='Wall',
                polygon=polygon.tolist(),
                thickness=width_rect,
                length=length_rect,
                height=cfg.default_wall_height,
            ))

        return walls

    def _extract_doors_windows(
        self, house, coef_width: float, coef_height: float
    ) -> Tuple[List[OpeningAnnotation], List[OpeningAnnotation]]:
        """从 house.icons 掩码提取门窗。"""
        cfg = self.config
        doors: List[OpeningAnnotation] = []
        windows: List[OpeningAnnotation] = []

        icons = house.icons

        # 提取门 (label=2)
        door_mask = (icons == DOOR_LABEL).astype(np.uint8)
        if door_mask.any():
            num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
                door_mask, connectivity=8
            )
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]

                if area < cfg.opening_min_area:
                    continue

                doors.append(OpeningAnnotation(
                    opening_type='Door',
                    bbox=[
                        x * coef_width, y * coef_height,
                        (x + w) * coef_width, (y + h) * coef_height,
                    ],
                    center=[centroids[i][0] * coef_width, centroids[i][1] * coef_height],
                    area=float(area) * coef_width * coef_height,
                    width=cfg.default_door_width,
                    height=cfg.default_door_height,
                ))

        # 提取窗 (label=1)
        window_mask = (icons == WINDOW_LABEL).astype(np.uint8)
        if window_mask.any():
            num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
                window_mask, connectivity=8
            )
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]

                if area < cfg.opening_min_area:
                    continue

                windows.append(OpeningAnnotation(
                    opening_type='Window',
                    bbox=[
                        x * coef_width, y * coef_height,
                        (x + w) * coef_width, (y + h) * coef_height,
                    ],
                    center=[centroids[i][0] * coef_width, centroids[i][1] * coef_height],
                    area=float(area) * coef_width * coef_height,
                    width=float(w) * coef_width,
                    height=cfg.default_window_height,
                    sill_height=cfg.default_sill_height,
                ))

        return doors, windows

    def _extract_railings(
        self, house, coef_width: float, coef_height: float
    ) -> List[StructuralAnnotation]:
        """从 house.walls 掩码提取栏杆。"""
        cfg = self.config
        railings: List[StructuralAnnotation] = []

        railing_mask = (house.walls == RAILING_LABEL).astype(np.uint8)
        if not railing_mask.any():
            return railings

        contours, _ = cv2.findContours(railing_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 20:
                continue

            epsilon = cfg.contour_approx_epsilon_ratio * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygon = approx.squeeze().astype(float)

            if polygon.ndim == 1:
                continue

            polygon[:, 0] *= coef_width
            polygon[:, 1] *= coef_height

            railings.append(StructuralAnnotation(
                element_type='Railing',
                polygon=polygon.tolist(),
                height=cfg.default_railing_height,
                thickness=cfg.default_railing_thickness,
            ))

        return railings


# ============================================================================
# PyTorch Dataset
# ============================================================================

if TORCH_AVAILABLE:
    class CubiCasaDataset(Dataset):
        """CubiCasa5k PyTorch 数据集包装器。"""

        def __init__(
            self,
            data_root: str,
            split: str = 'train',
            config: Optional[ParserConfig] = None,
            transform=None,
        ):
            self.parser = CubiCasaParser(data_root, config)
            self.samples = self.parser.load_split_file(split)
            self.transform = transform
            self.split = split

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> SampleData:
            sample = self.parser.load_sample(self.samples[idx])
            if self.transform:
                sample = self.transform(sample)
            return sample

        def get_sample_path(self, idx: int) -> str:
            return self.samples[idx]


# ============================================================================
# CLI 入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description='CubiCasa 解析器测试')
    ap.add_argument('--data_root', type=str, required=True, help='CubiCasa5k 路径')
    ap.add_argument('--split', type=str, default='train', help='数据划分')
    ap.add_argument('--num_samples', type=int, default=1, help='测试样本数')
    args = ap.parse_args()

    config = ParserConfig()
    parser_obj = CubiCasaParser(args.data_root, config)

    try:
        samples = parser_obj.load_split_file(args.split)
    except SplitFileError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("加载 %s 集: %d 个样本", args.split, len(samples))

    for i in range(min(args.num_samples, len(samples))):
        logger.info("=" * 60)
        logger.info("样本 %d: %s", i + 1, samples[i])

        try:
            sample = parser_obj.load_sample(samples[i])
        except (ImageLoadError, SVGParseError) as e:
            logger.error("加载失败: %s", e)
            continue

        logger.info("图像尺寸: %d x %d", sample.width, sample.height)
        logger.info("房间: %d  墙体: %d  门: %d  窗: %d",
                     len(sample.rooms), len(sample.walls),
                     len(sample.doors), len(sample.windows))

        if sample.rooms:
            room_types: Dict[str, int] = {}
            for room in sample.rooms:
                room_types[room.room_type] = room_types.get(room.room_type, 0) + 1
            for rt, count in room_types.items():
                logger.info("  %s: %d", rt, count)

    logger.info("测试完成")
