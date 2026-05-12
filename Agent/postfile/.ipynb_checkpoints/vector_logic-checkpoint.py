"""
vector_logic.py — 算法层：墙体矢量化（Section 2.4）

纯图像处理 / 几何运算，不依赖模型，不依赖 3D 库。
可单独测试、单独部署。

对外接口（post_service.py 调用）：
    vectorize_wall_mask(wall_mask, cfg) -> List[Tuple]
    compute_vectorization_iou(wall_boxes, gt_mask) -> float

内部流程（对应论文 Section 2.4 逐步骤）：
    Step 1  morphological_preprocessing
    Step 2  find_wall_angles
    Step 3-4 extract_wall_segments_at_angle
    Step 5  shrinking_algorithm  +  compute_iou_mask_box
    Step 6  resolve_overlapping_boxes
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from postprocess_config import VectorizationConfig, VECT_CFG

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Section 2.4 Step 1：形态学预处理
# ══════════════════════════════════════════════════════════════

def morphological_preprocessing(
    wall_mask: np.ndarray,
    cfg: VectorizationConfig = VECT_CFG,
) -> np.ndarray:
    """
    论文 Section 2.4 第一步：形态学预处理
    1. morphological opening  → 去除小噪点
    2. Gaussian blur          → 平滑边界
    3. morphological closing  → 填充墙体内部小洞
    """
    mask = wall_mask.astype(np.uint8)

    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_RECT, (cfg.morph_open_size, cfg.morph_open_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    mask_f = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), sigmaX=1.0)
    mask   = (mask_f > 0.5).astype(np.uint8)

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_RECT, (cfg.morph_close_size, cfg.morph_close_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


# ══════════════════════════════════════════════════════════════
# Section 2.4 Step 2：Canny + Hough 找墙体主要角度
# ══════════════════════════════════════════════════════════════

def find_wall_angles(
    wall_mask: np.ndarray,
    cfg: VectorizationConfig = VECT_CFG,
) -> List[float]:
    """
    论文 Section 2.4：
    Canny 边缘检测 + Hough 变换找墙体主要方向
    返回角度列表（度），始终包含 0.0（水平/垂直基准）
    """
    edges = cv2.Canny(
        (wall_mask * 255).astype(np.uint8),
        threshold1=50,
        threshold2=150,
    )

    lines = cv2.HoughLines(
        edges,
        rho=1,
        theta=np.deg2rad(cfg.hough_angle_resolution),
        threshold=50,
    )

    if lines is None:
        return [0.0]

    angles = []
    for line in lines:
        theta = np.rad2deg(line[0][1]) % 180
        if   5 < theta < 85:    angles.append(theta)
        elif 95 < theta < 175:  angles.append(theta - 90)

    if not angles:
        return [0.0]

    hist, bin_edges = np.histogram(angles, bins=180)
    top_bins    = np.argsort(hist)[::-1][:cfg.hough_n_angles]
    top_angles  = [bin_edges[b] for b in top_bins if hist[b] > 0]

    if 0.0 not in top_angles:
        top_angles = [0.0] + top_angles

    return top_angles[:cfg.hough_n_angles]


# ══════════════════════════════════════════════════════════════
# Section 2.4 Step 3-4：按角度提取单个墙体段
# ══════════════════════════════════════════════════════════════

def extract_wall_segments_at_angle(
    wall_mask: np.ndarray,
    angle_deg: float,
    cfg: VectorizationConfig = VECT_CFG,
) -> List[np.ndarray]:
    """
    论文 Section 2.4 Step 3-4：
    1. 旋转 mask，使该角度的墙体变为水平/垂直
    2. 形态学 opening 分离水平段和垂直段
    3. 轮廓提取得到单个墙体段
    4. 验证算法：跳过实际仍斜的轮廓（允许 ±10°）
    5. 反旋转轮廓回原始坐标系

    返回原始坐标系下的 contour 列表（np.int32）
    """
    h, w   = wall_mask.shape
    center = (w // 2, h // 2)

    M       = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        wall_mask.astype(np.uint8), M, (w, h),
        flags=cv2.INTER_NEAREST,
    )

    segments_rotated = []

    kernel_h = cv2.getStructuringElement(
        cv2.MORPH_RECT, (cfg.morph_horizontal_len, 1)
    )
    kernel_v = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, cfg.morph_vertical_len)
    )

    for component_mask in [
        cv2.morphologyEx(rotated, cv2.MORPH_OPEN, kernel_h),
        cv2.morphologyEx(rotated, cv2.MORPH_OPEN, kernel_v),
    ]:
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            if cv2.contourArea(cnt) < cfg.min_segment_area:
                continue
            rect_angle = abs(cv2.minAreaRect(cnt)[2])
            if 10 < rect_angle < 80:
                continue   # 验证算法：跳过真正倾斜的
            segments_rotated.append(cnt)

    M_inv = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    segments_original = []
    for cnt in segments_rotated:
        pts      = cnt.reshape(-1, 2).astype(np.float32)
        pts_back = cv2.transform(pts.reshape(1, -1, 2), M_inv)
        segments_original.append(
            pts_back.reshape(-1, 1, 2).astype(np.int32)
        )

    return segments_original


# ══════════════════════════════════════════════════════════════
# Section 2.4 Step 5：Shrinking 算法
# ══════════════════════════════════════════════════════════════

def compute_iou_mask_box(
    mask: np.ndarray,
    box: Tuple[int, int, int, int],
) -> float:
    """计算 binary mask 和矩形 box 的 IoU"""
    x1, y1, x2, y2 = box
    h, w = mask.shape
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    box_mask = np.zeros_like(mask)
    box_mask[y1:y2, x1:x2] = 1
    intersection = (mask & box_mask).sum()
    union        = (mask | box_mask).sum()
    return float(intersection) / (float(union) + 1e-8)


def shrinking_algorithm(
    wall_mask: np.ndarray,
    contour:   np.ndarray,
    cfg:       VectorizationConfig = VECT_CFG,
) -> Optional[Tuple[int, int, int, int]]:
    """
    论文 Section 2.4 Step 5：Shrinking 算法
    把不规则形状的墙体段收缩成最优矩形 bbox

    迭代策略：
      从 axis-aligned bounding box 出发，
      每轮从四个方向各收缩 1 像素，
      选择 IoU 最大的方向作为新 bbox，
      重复直到 IoU 达到阈值或 bbox 尺寸过小。
    """
    x, y, w_cnt, h_cnt = cv2.boundingRect(contour)
    box = (x, y, x + w_cnt, y + h_cnt)

    h, w = wall_mask.shape
    local_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(local_mask, [contour], -1, 1, -1)

    best_iou = compute_iou_mask_box(local_mask, box)
    improved = True

    while improved:
        improved = False
        x1, y1, x2, y2 = box

        if (x2 - x1) < cfg.shrink_min_size or (y2 - y1) < cfg.shrink_min_size:
            break

        for cand in [
            (x1 + 1, y1,     x2,     y2),   # 左缩
            (x1,     y1 + 1, x2,     y2),   # 上缩
            (x1,     y1,     x2 - 1, y2),   # 右缩
            (x1,     y1,     x2,     y2 - 1),  # 下缩
        ]:
            if cand[2] - cand[0] < cfg.shrink_min_size:
                continue
            if cand[3] - cand[1] < cfg.shrink_min_size:
                continue
            iou = compute_iou_mask_box(local_mask, cand)
            if iou > best_iou:
                best_iou = iou
                box      = cand
                improved = True

        if best_iou >= cfg.iou_threshold:
            break

    x1, y1, x2, y2 = box
    if (x2 - x1) < cfg.shrink_min_size or (y2 - y1) < cfg.shrink_min_size:
        return None
    return box


# ══════════════════════════════════════════════════════════════
# Section 2.4 Step 6：重叠处理
# ══════════════════════════════════════════════════════════════

def _box_area(box: Tuple) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _box_intersection(b1: Tuple, b2: Tuple) -> float:
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    return max(0.0, float(ix2 - ix1)) * max(0.0, float(iy2 - iy1))


def resolve_overlapping_boxes(
    boxes: List[Tuple],
    cfg:   VectorizationConfig = VECT_CFG,
) -> List[Tuple]:
    """
    论文 Section 2.4 Step 6：处理矢量化后重叠的 bbox

    规则：
      1. 小 box 被大 box 完全包含（>95%）→ 删除小 box
      2. 部分重叠 / overlap_ratio > overlap_thresh → 缩小面积较小的 box
    """
    if not boxes:
        return []

    boxes   = list(boxes)
    changed = True

    while changed:
        changed   = False
        to_remove = set()
        n         = len(boxes)

        for i in range(n):
            if i in to_remove:
                continue
            for j in range(i + 1, n):
                if j in to_remove:
                    continue

                inter  = _box_intersection(boxes[i], boxes[j])
                if inter == 0:
                    continue

                area_i = _box_area(boxes[i])
                area_j = _box_area(boxes[j])

                # 完全包含
                if inter >= area_i * 0.95:
                    to_remove.add(i)
                    changed = True
                    break
                elif inter >= area_j * 0.95:
                    to_remove.add(j)
                    changed = True
                    continue

                # 部分重叠
                overlap_ratio = inter / (min(area_i, area_j) + 1e-8)
                if overlap_ratio < cfg.overlap_thresh:
                    continue

                xi1, yi1, xi2, yi2 = boxes[i]
                xj1, yj1, xj2, yj2 = boxes[j]

                if area_i <= area_j:
                    # 缩小 i
                    if xi2 > xj1 and xi1 < xj1:
                        boxes[i] = (xi1, yi1, xj1, yi2)
                    elif xi1 < xj2 and xi2 > xj2:
                        boxes[i] = (xj2, yi1, xi2, yi2)
                    elif yi2 > yj1 and yi1 < yj1:
                        boxes[i] = (xi1, yi1, xi2, yj1)
                    elif yi1 < yj2 and yi2 > yj2:
                        boxes[i] = (xi1, yj2, xi2, yi2)
                else:
                    # 缩小 j
                    if xj2 > xi1 and xj1 < xi1:
                        boxes[j] = (xj1, yj1, xi1, yj2)
                    elif xj1 < xi2 and xj2 > xi2:
                        boxes[j] = (xi2, yj1, xj2, yj2)
                    elif yj2 > yi1 and yj1 < yi1:
                        boxes[j] = (xj1, yj1, xj2, yi1)
                    elif yj1 < yi2 and yj2 > yi2:
                        boxes[j] = (xj1, yi2, xj2, yj2)

                changed = True

        boxes = [b for i, b in enumerate(boxes) if i not in to_remove]

    return [b for b in boxes if _box_area(b) > 0]


# ══════════════════════════════════════════════════════════════
# 完整矢量化流程（Section 2.4 对外接口）
# ══════════════════════════════════════════════════════════════

def vectorize_wall_mask(
    wall_mask: np.ndarray,
    cfg:       VectorizationConfig = VECT_CFG,
) -> List[Tuple[int, int, int, int]]:
    """
    论文 Section 2.4 完整矢量化流程
    输入：binary wall mask (H, W) uint8
    输出：矩形 bbox 列表 [(x1, y1, x2, y2), ...]

    调用 post_service.py 时传入此函数。
    """
    # Step 1
    mask = morphological_preprocessing(wall_mask, cfg)
    logger.debug(f'预处理后 wall 像素: {mask.sum()}')

    # Step 2
    angles = find_wall_angles(mask, cfg)
    logger.debug(f'检测到角度: {angles}')

    # Step 3-4
    all_contours = []
    for angle in angles:
        all_contours.extend(extract_wall_segments_at_angle(mask, angle, cfg))
    logger.debug(f'提取到轮廓数: {len(all_contours)}')

    # Step 5
    wall_boxes = []
    for cnt in all_contours:
        box = shrinking_algorithm(mask, cnt, cfg)
        if box is not None:
            wall_boxes.append(box)
    logger.debug(f'Shrinking 后 bbox 数: {len(wall_boxes)}')

    # Step 6
    wall_boxes = resolve_overlapping_boxes(wall_boxes, cfg)
    logger.debug(f'去重叠后 bbox 数: {len(wall_boxes)}')

    return wall_boxes


# ══════════════════════════════════════════════════════════════
# 评估工具：矢量化 IoU（论文 Table 1 的 IoU vect 指标）
# ══════════════════════════════════════════════════════════════

def compute_vectorization_iou(
    wall_boxes: List[Tuple],
    gt_mask:    np.ndarray,
) -> float:
    """
    论文 Table 1：IoU vect 指标
    把矢量化 bbox 列表转成 binary mask，和 GT mask 计算 IoU
    论文结果：Ours IoU vect = 0.80
    """
    h, w = gt_mask.shape
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    for box in wall_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        pred_mask[y1:y2, x1:x2] = 1

    intersection = (pred_mask & gt_mask.astype(np.uint8)).sum()
    union        = (pred_mask | gt_mask.astype(np.uint8)).sum()
    return float(intersection) / (float(union) + 1e-8)


def wall_boxes_to_mask(
    wall_boxes: List[Tuple],
    shape:      Tuple[int, int],
) -> np.ndarray:
    """辅助工具：把 wall_boxes 转成 binary mask（用于差异图可视化）"""
    h, w = shape
    m    = np.zeros((h, w), dtype=np.uint8)
    for b in wall_boxes:
        x1, y1, x2, y2 = [int(v) for v in b]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        m[y1:y2, x1:x2] = 1
    return m
