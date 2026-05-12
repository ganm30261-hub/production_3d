# tools/sam2_refine.py
import os
from typing import Optional, Tuple

import cv2
import numpy as np

from config import logger, DEVICE, PseudoLabelConfig

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False
    logger.warning('[!] SAM2 未安装: pip install segment-anything-2')


# ══════════════════════════════════════════════════════════════
# 点采样
# ══════════════════════════════════════════════════════════════

def sample_points_from_mask(
    mask:     np.ndarray,
    n_pos:    int = 5,
    n_neg:    int = 3,
    erode_px: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 binary mask 采样正负点，供 SAM2 Point Prompt 使用。

    正点策略：先对前景做形态学侵蚀（erode_px 像素），
              再从侵蚀后的区域随机采样，避免采到噪声边界附近。
    负点策略：从背景区域（mask==0）随机采样。

    参数:
        mask     : (H, W) uint8，0=背景 1=前景
        n_pos    : 正点数量
        n_neg    : 负点数量
        erode_px : 侵蚀半径（像素）

    返回:
        points : (N, 2) float32  [x, y]  SAM2 约定 x 在前
        labels : (N,)   int32    1=正点 0=负点
    """
    points, labels = [], []

    # ── 正点：从侵蚀后的前景采样 ──
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
    )
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    pos_yx = np.argwhere(eroded > 0)   # (N, 2) [row, col]

    if len(pos_yx) >= n_pos:
        chosen = pos_yx[np.random.choice(len(pos_yx), n_pos, replace=False)]
    elif len(pos_yx) > 0:
        chosen = pos_yx[np.random.choice(len(pos_yx), n_pos, replace=True)]
    else:
        chosen = np.zeros((0, 2), dtype=int)

    for r, c in chosen:
        points.append([c, r])   # SAM2 用 [x, y] = [col, row]
        labels.append(1)

    # ── 负点：从背景采样 ──
    neg_yx = np.argwhere(mask == 0)
    if len(neg_yx) >= n_neg:
        chosen = neg_yx[np.random.choice(len(neg_yx), n_neg, replace=False)]
        for r, c in chosen:
            points.append([c, r])
            labels.append(0)

    if not points:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.int32)

    return (
        np.array(points, dtype=np.float32),
        np.array(labels, dtype=np.int32),
    )


# ══════════════════════════════════════════════════════════════
# SAM2 精化
# ══════════════════════════════════════════════════════════════

def refine_mask_with_sam2(
    image_rgb:    np.ndarray,
    initial_mask: np.ndarray,
    sam2_predictor,
    cfg:          PseudoLabelConfig,
) -> np.ndarray:
    """
    用 SAM2 Point Prompt 精化初始 mask，改善边界清晰度。

    流程：
        1. set_image  → SAM2 编码图像特征（一张图只编码一次）
        2. 从 initial_mask 采样正负点
        3. predict    → 多个 mask 候选（multimask_output=True）
        4. 在 score >= sam2_score_thresh 的候选里，
           选与 initial_mask IoU 最大的（语义一致性优先于 SAM2 置信度）

    参数:
        image_rgb    : (H, W, 3) uint8 RGB
        initial_mask : (H, W)   uint8  0/1，DINOv2+LoRA 的初始输出
        sam2_predictor : SAM2ImagePredictor 实例
        cfg          : PseudoLabelConfig

    返回:
        refined_mask : (H, W) uint8  0/1
    """
    if not HAS_SAM2:
        logger.warning('SAM2 未安装，跳过精化，返回初始 mask')
        return initial_mask

    sam2_predictor.set_image(image_rgb)

    points, point_labels = sample_points_from_mask(
        initial_mask,
        n_pos = cfg.sam2_n_pos_points,
        n_neg = cfg.sam2_n_neg_points,
    )
    if len(points) == 0:
        logger.warning('无法从 mask 采样点，返回初始 mask')
        return initial_mask

    masks, scores, _ = sam2_predictor.predict(
        point_coords     = points,
        point_labels     = point_labels,
        multimask_output = True,   # 输出多个候选
    )
    # masks : (N_candidates, H, W)
    # scores: (N_candidates,)

    # 在置信度达标的候选中选 IoU 最大的
    init_bool            = initial_mask.astype(bool)
    best_idx, best_iou_v = 0, -1.0

    for i, (m, s) in enumerate(zip(masks, scores)):
        if s < cfg.sam2_score_thresh:
            continue
        m_bool  = m.astype(bool)
        inter   = (m_bool & init_bool).sum()
        union   = (m_bool | init_bool).sum()
        iou_val = inter / (union + 1e-8)
        if iou_val > best_iou_v:
            best_iou_v = iou_val
            best_idx   = i

    refined = masks[best_idx].astype(np.uint8)
    logger.debug(
        f'SAM2 精化完成  '
        f'init_pixels={initial_mask.sum()}  '
        f'refined_pixels={refined.sum()}  '
        f'iou_with_init={best_iou_v:.3f}'
    )
    return refined


# ══════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════

def load_sam2_predictor(cfg: PseudoLabelConfig) -> Optional['SAM2ImagePredictor']:
    """
    懒加载 SAM2 ImagePredictor，避免在不需要时占用显存。

    权重下载（如尚未下载）：
        wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
    """
    if not HAS_SAM2:
        logger.error('SAM2 未安装: pip install segment-anything-2')
        return None

    if not cfg.sam2_ckpt or not os.path.exists(cfg.sam2_ckpt):
        logger.error(
            f'SAM2 权重不存在: {cfg.sam2_ckpt}\n'
            f'下载命令: wget https://dl.fbaipublicfiles.com/'
            f'segment_anything_2/072824/sam2_hiera_large.pt'
        )
        return None

    sam2_model = build_sam2(cfg.sam2_cfg, cfg.sam2_ckpt, device=DEVICE)
    predictor  = SAM2ImagePredictor(sam2_model)
    logger.info('✓ SAM2 ImagePredictor 加载完成')
    return predictor
