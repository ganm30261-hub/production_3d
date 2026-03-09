"""
预处理模块（生产级）

功能：
1. 裁剪到墙体区域 (Crop to Wall Mask)
2. 滑动窗口 (Sliding Window)
3. 数据增强 (Rotation, Scale, Flip, Color Jitter)
4. Preprocessor 整合类支持 train / eval 模式

使用方法：
    from data.preprocessing import Preprocessor, PreprocessConfig

    preprocessor = Preprocessor(PreprocessConfig(), mode='train')
    result = preprocessor.process(image, walls_mask, icons_mask)
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import cv2
import numpy as np

from .exceptions import PreprocessingError

logger = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class CroppingConfig:
    """裁剪配置"""
    enable: bool = True
    padding: int = 20
    wall_label: int = 2
    min_size: int = 100


@dataclass
class SlidingWindowConfig:
    """滑动窗口配置"""
    enable: bool = True
    window_size: Tuple[int, int] = (512, 512)
    stride: int = 256
    pad_mode: str = 'constant'
    pad_value: int = 255


@dataclass
class AugmentationConfig:
    """数据增强配置"""
    enable: bool = True

    # 旋转
    enable_rotation: bool = True
    rotation_range: Tuple[float, float] = (-15.0, 15.0)
    rotation_prob: float = 0.5

    # 缩放
    enable_scale: bool = True
    scale_range: Tuple[float, float] = (0.9, 1.1)
    scale_prob: float = 0.5

    # 翻转
    enable_flip: bool = True
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.3

    # 颜色抖动
    enable_color_jitter: bool = False
    brightness_range: Tuple[float, float] = (0.9, 1.1)
    contrast_range: Tuple[float, float] = (0.9, 1.1)

    # 填充值
    border_value: Tuple[int, int, int] = (255, 255, 255)
    mask_border_value: int = 0


@dataclass
class PreprocessConfig:
    """完整预处理配置"""
    cropping: CroppingConfig = field(default_factory=CroppingConfig)
    sliding_window: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    seed: Optional[int] = None


# ============================================================================
# 裁剪到墙体区域
# ============================================================================

class WallCropper:
    """
    裁剪到墙体区域。

    去除图像中未标注的区域（标题、文字说明等），
    只保留有实际建筑标注的部分。
    """

    def __init__(self, config: CroppingConfig):
        self.config = config

    def crop(
        self,
        image: np.ndarray,
        walls_mask: np.ndarray,
        other_masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        裁剪图像和掩码到墙体标注区域。

        论文 Section 3:
        "We address this by cropping the images and the corresponding labels
        to the size of the wall masks on the ground truth, leaving out in most
        cases the elements that have not been annotated."

        裁剪边界由 walls_mask 中 wall_label (默认2) 像素的范围决定，
        而非所有非零像素。这样只保留有墙壁标注覆盖的区域，
        排除"图像中可见但未标注"的部分，提升训练数据质量。

        Args:
            image: (H, W, C) BGR 图像
            walls_mask: (H, W) 多类别语义掩码 (house.walls)
            other_masks: 其他需要同步裁剪的掩码 (如 icons)

        Returns:
            包含 image, walls_mask, other_masks, crop_bbox, original_size 的字典
        """
        if not self.config.enable:
            return self._passthrough(image, walls_mask, other_masks)

        # 论文关键：只用 wall_label 像素确定裁剪边界
        wall_pixels = (walls_mask == self.config.wall_label)
        if not wall_pixels.any():
            logger.debug("未找到墙壁标注 (label=%d)，跳过裁剪", self.config.wall_label)
            return self._passthrough(image, walls_mask, other_masks)

        # 计算墙壁标注区域边界
        rows = np.any(wall_pixels, axis=1)
        cols = np.any(wall_pixels, axis=0)
        y_min, y_max = int(np.where(rows)[0][[0, -1]][0]), int(np.where(rows)[0][[0, -1]][1])
        x_min, x_max = int(np.where(cols)[0][[0, -1]][0]), int(np.where(cols)[0][[0, -1]][1])

        h, w = image.shape[:2]
        pad = self.config.padding
        y_min = max(0, y_min - pad)
        y_max = min(h, y_max + pad + 1)
        x_min = max(0, x_min - pad)
        x_max = min(w, x_max + pad + 1)

        # 保证最小尺寸
        y_min, y_max = self._ensure_min_size(y_min, y_max, h, self.config.min_size)
        x_min, x_max = self._ensure_min_size(x_min, x_max, w, self.config.min_size)

        # 执行裁剪
        cropped_image = image[y_min:y_max, x_min:x_max].copy()
        cropped_walls = walls_mask[y_min:y_max, x_min:x_max].copy()

        cropped_others = {}
        if other_masks:
            for name, mask in other_masks.items():
                cropped_others[name] = mask[y_min:y_max, x_min:x_max].copy()

        logger.debug("裁剪: (%d,%d,%d,%d) → 尺寸 %dx%d",
                      y_min, x_min, y_max, x_max,
                      y_max - y_min, x_max - x_min)

        return {
            'image': cropped_image,
            'walls_mask': cropped_walls,
            'other_masks': cropped_others,
            'crop_bbox': (y_min, x_min, y_max, x_max),
            'original_size': (h, w),
        }

    @staticmethod
    def _passthrough(image, walls_mask, other_masks):
        return {
            'image': image,
            'walls_mask': walls_mask,
            'other_masks': other_masks or {},
            'crop_bbox': None,
            'original_size': image.shape[:2],
        }

    @staticmethod
    def _ensure_min_size(lo: int, hi: int, limit: int, min_size: int) -> Tuple[int, int]:
        if hi - lo < min_size:
            center = (lo + hi) // 2
            lo = max(0, center - min_size // 2)
            hi = min(limit, lo + min_size)
        return lo, hi


# ============================================================================
# 滑动窗口
# ============================================================================

class SlidingWindowProcessor:
    """
    滑动窗口处理器。

    将大尺寸图像切分为固定大小的 patch，
    支持推理时拼回原图（带重叠区域平均融合）。
    """

    def __init__(self, config: SlidingWindowConfig):
        self.config = config

    def extract_patches(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        提取 patch。

        Returns:
            patches, mask_patches, positions, original_size, num_patches
        """
        if not self.config.enable:
            return {
                'patches': [image],
                'mask_patches': [mask] if mask is not None else None,
                'positions': [(0, 0)],
                'original_size': image.shape[:2],
                'num_patches': (1, 1),
            }

        h, w = image.shape[:2]
        win_h, win_w = self.config.window_size
        stride = self.config.stride

        # 如果图像小于窗口，先填充
        image, mask, h, w = self._pad_if_needed(image, mask, win_h, win_w)

        num_h = max(1, (h - win_h) // stride + 1)
        num_w = max(1, (w - win_w) // stride + 1)

        patches: List[np.ndarray] = []
        mask_patches: List[np.ndarray] = []
        positions: List[Tuple[int, int]] = []

        for i in range(num_h):
            for j in range(num_w):
                y = min(i * stride, h - win_h)
                x = min(j * stride, w - win_w)

                patches.append(image[y:y + win_h, x:x + win_w])
                positions.append((y, x))

                if mask is not None:
                    mask_patches.append(mask[y:y + win_h, x:x + win_w])

        logger.debug("滑动窗口: %dx%d patches (stride=%d)", num_h, num_w, stride)

        return {
            'patches': patches,
            'mask_patches': mask_patches if mask is not None else None,
            'positions': positions,
            'original_size': (h, w),
            'num_patches': (num_h, num_w),
        }

    def merge_patches(
        self,
        patches: List[np.ndarray],
        positions: List[Tuple[int, int]],
        output_size: Tuple[int, int],
        mode: str = 'average',
    ) -> np.ndarray:
        """合并 patch 为完整图像（重叠区域平均融合）。"""
        h, w = output_size
        win_h, win_w = self.config.window_size

        sample = patches[0]
        if sample.ndim == 3:
            output = np.zeros((h, w, sample.shape[2]), dtype=np.float32)
        else:
            output = np.zeros((h, w), dtype=np.float32)

        count = np.zeros((h, w), dtype=np.float32)

        for patch, (y, x) in zip(patches, positions):
            y_end = min(y + win_h, h)
            x_end = min(x + win_w, w)
            ph, pw = y_end - y, x_end - x

            if output.ndim == 3:
                output[y:y_end, x:x_end] += patch[:ph, :pw]
            else:
                output[y:y_end, x:x_end] += patch[:ph, :pw]
            count[y:y_end, x:x_end] += 1

        count = np.maximum(count, 1)
        if output.ndim == 3:
            output /= count[:, :, np.newaxis]
        else:
            output /= count

        return output

    def _pad_if_needed(
        self, image: np.ndarray, mask: Optional[np.ndarray], win_h: int, win_w: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], int, int]:
        h, w = image.shape[:2]
        if h >= win_h and w >= win_w:
            return image, mask, h, w

        pad_h = max(0, win_h - h)
        pad_w = max(0, win_w - w)
        pv = self.config.pad_value

        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w,
            borderType=cv2.BORDER_CONSTANT,
            value=pv if image.ndim == 2 else [pv] * 3,
        )
        if mask is not None:
            mask = cv2.copyMakeBorder(
                mask, 0, pad_h, 0, pad_w,
                borderType=cv2.BORDER_CONSTANT, value=0,
            )
        return image, mask, image.shape[0], image.shape[1]


# ============================================================================
# 数据增强
# ============================================================================

class DataAugmentor:
    """
    数据增强器。

    同步变换图像、掩码和边界框，保证标注一致性。
    """

    def __init__(self, config: AugmentationConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        image: np.ndarray,
        masks: Optional[Dict[str, np.ndarray]] = None,
        bboxes: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        应用随机数据增强。

        Returns:
            image, masks, bboxes, transforms (记录应用了哪些变换)
        """
        if not self.config.enable:
            return {'image': image, 'masks': masks, 'bboxes': bboxes, 'transforms': []}

        image = image.copy()
        masks = {k: v.copy() for k, v in masks.items()} if masks else {}
        bboxes = [list(b) for b in bboxes] if bboxes else []
        transforms: List[str] = []

        cfg = self.config

        if cfg.enable_rotation and self.rng.random() < cfg.rotation_prob:
            angle = float(self.rng.uniform(*cfg.rotation_range))
            image, masks, bboxes = self._rotate(image, masks, bboxes, angle)
            transforms.append(f'rotate({angle:.1f})')

        if cfg.enable_scale and self.rng.random() < cfg.scale_prob:
            scale = float(self.rng.uniform(*cfg.scale_range))
            image, masks, bboxes = self._scale(image, masks, bboxes, scale)
            transforms.append(f'scale({scale:.2f})')

        if cfg.enable_flip and self.rng.random() < cfg.horizontal_flip_prob:
            image, masks, bboxes = self._flip_horizontal(image, masks, bboxes)
            transforms.append('flip_h')

        if cfg.enable_flip and self.rng.random() < cfg.vertical_flip_prob:
            image, masks, bboxes = self._flip_vertical(image, masks, bboxes)
            transforms.append('flip_v')

        if cfg.enable_color_jitter:
            image = self._color_jitter(image)
            transforms.append('color_jitter')

        return {'image': image, 'masks': masks, 'bboxes': bboxes, 'transforms': transforms}

    # ------------------------------------------------------------------
    # 变换实现
    # ------------------------------------------------------------------

    def _rotate(self, image, masks, bboxes, angle):
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        image = cv2.warpAffine(
            image, M, (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.config.border_value,
        )
        for name in masks:
            masks[name] = cv2.warpAffine(
                masks[name], M, (new_w, new_h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.config.mask_border_value,
                flags=cv2.INTER_NEAREST,
            )
        if bboxes:
            bboxes = self._rotate_bboxes(bboxes, M, new_w, new_h)
        return image, masks, bboxes

    @staticmethod
    def _rotate_bboxes(bboxes, M, new_w, new_h):
        new_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4]
            corners = np.array([
                [x1, y1, 1], [x2, y1, 1],
                [x2, y2, 1], [x1, y2, 1],
            ])
            rotated = corners @ M.T
            nx1 = max(0, float(rotated[:, 0].min()))
            ny1 = max(0, float(rotated[:, 1].min()))
            nx2 = min(new_w, float(rotated[:, 0].max()))
            ny2 = min(new_h, float(rotated[:, 1].max()))
            if nx2 > nx1 and ny2 > ny1:
                new_bboxes.append([nx1, ny1, nx2, ny2] + list(bbox[4:]))
        return new_bboxes

    def _scale(self, image, masks, bboxes, scale):
        h, w = image.shape[:2]
        nh, nw = int(h * scale), int(w * scale)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        for name in masks:
            masks[name] = cv2.resize(masks[name], (nw, nh), interpolation=cv2.INTER_NEAREST)
        if bboxes:
            bboxes = [[b[0]*scale, b[1]*scale, b[2]*scale, b[3]*scale] + list(b[4:]) for b in bboxes]
        return image, masks, bboxes

    def _flip_horizontal(self, image, masks, bboxes):
        w = image.shape[1]
        image = cv2.flip(image, 1)
        for name in masks:
            masks[name] = cv2.flip(masks[name], 1)
        if bboxes:
            bboxes = [[w - b[2], b[1], w - b[0], b[3]] + list(b[4:]) for b in bboxes]
        return image, masks, bboxes

    def _flip_vertical(self, image, masks, bboxes):
        h = image.shape[0]
        image = cv2.flip(image, 0)
        for name in masks:
            masks[name] = cv2.flip(masks[name], 0)
        if bboxes:
            bboxes = [[b[0], h - b[3], b[2], h - b[1]] + list(b[4:]) for b in bboxes]
        return image, masks, bboxes

    def _color_jitter(self, image):
        brightness = self.rng.uniform(*self.config.brightness_range)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        contrast = self.rng.uniform(*self.config.contrast_range)
        mean = image.mean()
        image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
        return image


# ============================================================================
# 完整预处理器
# ============================================================================

class Preprocessor:
    """
    整合裁剪 → 增强 → 滑动窗口的完整预处理器。
    """

    def __init__(self, config: Optional[PreprocessConfig] = None, mode: str = 'train'):
        """
        Args:
            config: 预处理配置
            mode: 'train' (启用增强) 或 'eval' (仅裁剪+窗口)
        """
        if mode not in ('train', 'eval'):
            raise PreprocessingError(f"mode 必须为 'train' 或 'eval'，收到: {mode}")

        self.config = config or PreprocessConfig()
        self.mode = mode

        self.cropper = WallCropper(self.config.cropping)
        self.sliding_window = SlidingWindowProcessor(self.config.sliding_window)
        self.augmentor = DataAugmentor(self.config.augmentation, seed=self.config.seed)

    def process(
        self,
        image: np.ndarray,
        walls_mask: Optional[np.ndarray] = None,
        icons_mask: Optional[np.ndarray] = None,
        bboxes: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        处理单张图像。

        流程: 裁剪 → 数据增强(train) → 返回结果
        滑动窗口通过 extract_patches / merge_patches 单独调用。

        Returns:
            image, walls_mask, icons_mask, bboxes, crop_bbox,
            original_size, processed_size, transforms
        """
        result: Dict[str, Any] = {
            'original_image': image,
            'original_size': image.shape[:2],
        }

        # 1. 裁剪
        if walls_mask is not None:
            other = {'icons': icons_mask} if icons_mask is not None else None
            crop_out = self.cropper.crop(image, walls_mask, other)
            image = crop_out['image']
            walls_mask = crop_out['walls_mask']
            if crop_out['other_masks']:
                icons_mask = crop_out['other_masks'].get('icons')
            result['crop_bbox'] = crop_out['crop_bbox']

        # 2. 数据增强（仅 train）
        if self.mode == 'train' and self.config.augmentation.enable:
            masks_dict: Dict[str, np.ndarray] = {}
            if walls_mask is not None:
                masks_dict['walls'] = walls_mask
            if icons_mask is not None:
                masks_dict['icons'] = icons_mask

            aug_out = self.augmentor(image, masks_dict, bboxes)
            image = aug_out['image']
            walls_mask = aug_out['masks'].get('walls')
            icons_mask = aug_out['masks'].get('icons')
            bboxes = aug_out['bboxes']
            result['transforms'] = aug_out['transforms']

        result.update({
            'image': image,
            'walls_mask': walls_mask,
            'icons_mask': icons_mask,
            'bboxes': bboxes,
            'processed_size': image.shape[:2],
        })
        return result

    def extract_patches(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """提取 patch（通常在推理时使用）。"""
        return self.sliding_window.extract_patches(image, mask)

    def merge_patches(
        self,
        patches: List[np.ndarray],
        positions: List[Tuple[int, int]],
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """合并 patch（通常在推理时使用）。"""
        return self.sliding_window.merge_patches(patches, positions, output_size)


# ============================================================================
# CLI 测试
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("预处理模块测试")

    # 构造测试数据
    image = np.random.randint(0, 255, (1000, 1200, 3), dtype=np.uint8)
    walls_mask = np.zeros((1000, 1200), dtype=np.uint8)
    walls_mask[200:800, 300:900] = 2
    icons_mask = np.zeros((1000, 1200), dtype=np.uint8)
    icons_mask[300:350, 400:450] = 1
    icons_mask[500:550, 600:650] = 2

    config = PreprocessConfig()
    preprocessor = Preprocessor(config, mode='train')

    result = preprocessor.process(image, walls_mask, icons_mask)
    logger.info("原始尺寸: %s", result['original_size'])
    logger.info("处理后尺寸: %s", result['processed_size'])
    logger.info("裁剪边界: %s", result.get('crop_bbox'))
    logger.info("应用的变换: %s", result.get('transforms', []))

    patches_result = preprocessor.extract_patches(result['image'])
    logger.info("Patch 数量: %d", len(patches_result['patches']))
    logger.info("Patch 尺寸: %s", patches_result['patches'][0].shape)

    logger.info("测试完成")
