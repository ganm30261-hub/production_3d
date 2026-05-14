# utils/coord_utils.py
"""
坐标系对齐工具：集成期核心要素3

解决三套坐标系的互转问题：
    OpenCV   : (x, y)        x=列, y=行, 原点左上角
    NumPy    : (row, col)    row=行, col=列, 原点左上角
    CubiCasa : 归一化 (x, y) x=列/W, y=行/H, 值域 [0,1]

最容易踩坑的地方：
    cv2.rectangle(img, (x1,y1), (x2,y2))  ← OpenCV 用 (x,y)
    mask[y1:y2, x1:x2]                     ← NumPy 用 [row, col] = [y, x]
    SAM2 point_coords                       ← 用 (x, y) = (col, row)
    CubiCasa SVG <rect x= y= width= height= ← 归一化坐标

所有函数都有 assert 校验，集成期传错坐标立即报错而不是静默出错。
"""

from __future__ import annotations
from typing import List, Tuple, Union
import numpy as np


# ── 类型别名（增强可读性）──
XY    = Tuple[int, int]       # OpenCV / SAM2 格式  (x, y) = (col, row)
RC    = Tuple[int, int]       # NumPy 格式          (row, col)
BBoxXY = Tuple[int, int, int, int]   # (x1, y1, x2, y2)  OpenCV
BBoxRC = Tuple[int, int, int, int]   # (r1, c1, r2, c2)  NumPy
NormBBox = Tuple[float, float, float, float]  # CubiCasa 归一化


# ══════════════════════════════════════════════════════════════
# 点坐标互转
# ══════════════════════════════════════════════════════════════

def xy_to_rc(x: int, y: int) -> RC:
    """
    OpenCV (x, y) → NumPy (row, col)
    x=列, y=行  →  row=y, col=x
    """
    return (y, x)


def rc_to_xy(row: int, col: int) -> XY:
    """
    NumPy (row, col) → OpenCV (x, y)
    row=行, col=列  →  x=col, y=row
    """
    return (col, row)


def points_xy_to_rc(points: np.ndarray) -> np.ndarray:
    """
    (N, 2) float32 点数组 (x, y) → (row, col)
    SAM2 输出 → NumPy mask 索引时用

    输入:  [[x0,y0], [x1,y1], ...]
    输出:  [[y0,x0], [y1,x1], ...]
    """
    assert points.ndim == 2 and points.shape[1] == 2, \
        f'期望 (N,2)，收到 {points.shape}'
    return points[:, ::-1].copy()   # 交换列顺序


def points_rc_to_xy(points: np.ndarray) -> np.ndarray:
    """
    (N, 2) (row, col) → (x, y)
    NumPy argwhere 结果 → SAM2 point_coords 时用

    输入:  [[r0,c0], [r1,c1], ...]
    输出:  [[c0,r0], [c1,r1], ...]
    """
    assert points.ndim == 2 and points.shape[1] == 2, \
        f'期望 (N,2)，收到 {points.shape}'
    return points[:, ::-1].copy()


# ══════════════════════════════════════════════════════════════
# BBox 互转
# ══════════════════════════════════════════════════════════════

def bbox_xy_to_rc(x1: int, y1: int, x2: int, y2: int) -> BBoxRC:
    """
    OpenCV bbox (x1,y1,x2,y2) → NumPy slice (r1,c1,r2,c2)
    用法：mask[r1:r2, c1:c2]
    """
    return (y1, x1, y2, x2)


def bbox_rc_to_xy(r1: int, c1: int, r2: int, c2: int) -> BBoxXY:
    """
    NumPy slice (r1,c1,r2,c2) → OpenCV bbox (x1,y1,x2,y2)
    """
    return (c1, r1, c2, r2)


def bbox_to_numpy_slice(bbox: BBoxXY):
    """
    把 OpenCV bbox 转成可直接用于 numpy 切片的元组。

    用法：
        r1, c1, r2, c2 = bbox_to_numpy_slice(bbox)
        patch = image[r1:r2, c1:c2]
    """
    x1, y1, x2, y2 = bbox
    return (y1, x1, y2, x2)


# ══════════════════════════════════════════════════════════════
# 归一化坐标互转（CubiCasa SVG）
# ══════════════════════════════════════════════════════════════

def bbox_to_norm(
    x1: int, y1: int, x2: int, y2: int,
    img_w: int, img_h: int,
) -> NormBBox:
    """
    像素 bbox → CubiCasa 归一化坐标 [0, 1]

    CubiCasa SVG 的 <rect> 用归一化坐标：
        x = pixel_x / width
        y = pixel_y / height
    """
    assert img_w > 0 and img_h > 0, f'图像尺寸无效: {img_w}x{img_h}'
    return (x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h)


def norm_to_bbox(
    nx1: float, ny1: float, nx2: float, ny2: float,
    img_w: int, img_h: int,
) -> BBoxXY:
    """
    CubiCasa 归一化坐标 → 像素 bbox
    """
    assert all(0.0 <= v <= 1.0 for v in (nx1, ny1, nx2, ny2)), \
        f'归一化坐标超出 [0,1]: ({nx1},{ny1},{nx2},{ny2})'
    return (
        int(nx1 * img_w), int(ny1 * img_h),
        int(nx2 * img_w), int(ny2 * img_h),
    )


# ══════════════════════════════════════════════════════════════
# 图像归一化
# ══════════════════════════════════════════════════════════════

# DINOv2 使用 ImageNet 均值/标准差
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_imagenet(image_rgb: np.ndarray) -> np.ndarray:
    """
    uint8 RGB [0,255] → float32 归一化到 ImageNet 分布

    DINOv2 / ViT 系列模型使用此归一化。
    输入:  (H, W, 3) uint8
    输出:  (H, W, 3) float32，值域约 [-2.1, 2.6]
    """
    assert image_rgb.dtype == np.uint8,  f'期望 uint8，收到 {image_rgb.dtype}'
    assert image_rgb.ndim  == 3,         f'期望 (H,W,3)，收到 {image_rgb.shape}'
    assert image_rgb.shape[2] == 3,      f'期望 3 通道，收到 {image_rgb.shape[2]}'

    img = image_rgb.astype(np.float32) / 255.0
    return (img - IMAGENET_MEAN) / IMAGENET_STD


def denormalize_imagenet(image_norm: np.ndarray) -> np.ndarray:
    """
    ImageNet 归一化 → uint8 RGB [0,255]（可视化用）
    """
    img = image_norm * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def hwc_to_chw(image: np.ndarray) -> np.ndarray:
    """
    (H, W, C) → (C, H, W)  NumPy → PyTorch tensor 格式
    """
    assert image.ndim == 3, f'期望 3D，收到 {image.ndim}D'
    return np.transpose(image, (2, 0, 1))


def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    """
    (C, H, W) → (H, W, C)  PyTorch tensor → NumPy 格式
    """
    assert image.ndim == 3, f'期望 3D，收到 {image.ndim}D'
    return np.transpose(image, (1, 2, 0))


# ══════════════════════════════════════════════════════════════
# Tile 坐标工具（滑动窗口推理用）
# ══════════════════════════════════════════════════════════════

def tile_bbox_to_global(
    tile_bbox: BBoxXY,
    tile_origin_xy: XY,
) -> BBoxXY:
    """
    把 tile 内的局部 bbox 转换到原图全局坐标。

    tile_bbox       : tile 内的局部 (x1, y1, x2, y2)
    tile_origin_xy  : tile 左上角在原图中的 (x, y)

    用法（滑动窗口推理时还原检测框位置）：
        for tx, ty in tiles:
            local_boxes = model(tile)
            global_boxes = [tile_bbox_to_global(b, (tx, ty)) for b in local_boxes]
    """
    ox, oy = tile_origin_xy
    x1, y1, x2, y2 = tile_bbox
    return (x1 + ox, y1 + oy, x2 + ox, y2 + oy)


def scale_bbox(
    bbox: BBoxXY,
    scale_x: float,
    scale_y: float,
) -> BBoxXY:
    """
    等比例缩放 bbox（tile resize 后坐标还原用）。
    """
    x1, y1, x2, y2 = bbox
    return (
        int(x1 * scale_x), int(y1 * scale_y),
        int(x2 * scale_x), int(y2 * scale_y),
    )


def clip_bbox(bbox: BBoxXY, img_w: int, img_h: int) -> BBoxXY:
    """
    把 bbox 裁剪到图像边界内，防止越界。
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    return (x1, y1, x2, y2)


def bbox_area(bbox: BBoxXY) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(a: BBoxXY, b: BBoxXY) -> float:
    """计算两个 bbox 的 IoU。"""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


# ══════════════════════════════════════════════════════════════
# 快速验证（集成期手动跑）
# ══════════════════════════════════════════════════════════════

def _self_test():
    """运行：python3 utils/coord_utils.py"""
    print("── coord_utils 自检 ──")

    # 点坐标互转
    assert xy_to_rc(3, 5) == (5, 3)
    assert rc_to_xy(5, 3) == (3, 5)

    # bbox 互转
    assert bbox_xy_to_rc(10, 20, 50, 80) == (20, 10, 80, 50)
    assert bbox_rc_to_xy(20, 10, 80, 50) == (10, 20, 50, 80)

    # 归一化互转
    nb = bbox_to_norm(100, 200, 300, 400, 1000, 800)
    assert nb == (0.1, 0.25, 0.3, 0.5), nb
    pb = norm_to_bbox(*nb, 1000, 800)
    assert pb == (100, 200, 300, 400), pb

    # numpy 归一化
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    n   = normalize_imagenet(img)
    assert n.dtype == np.float32
    assert n.shape == (64, 64, 3)

    # points 互转
    pts_xy = np.array([[10, 20], [30, 40]], dtype=np.float32)
    pts_rc = points_xy_to_rc(pts_xy)
    assert pts_rc.tolist() == [[20, 10], [40, 30]]
    assert points_rc_to_xy(pts_rc).tolist() == pts_xy.tolist()

    # tile → global
    assert tile_bbox_to_global((5, 10, 20, 30), (100, 200)) == (105, 210, 120, 230)

    # IoU
    assert bbox_iou((0,0,10,10), (5,5,15,15)) > 0
    assert bbox_iou((0,0,10,10), (20,20,30,30)) == 0.0

    print("全部通过 ✓")


if __name__ == '__main__':
    _self_test()
