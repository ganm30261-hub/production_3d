# skeleon/tool_validator.py
"""
工具级容错：集成期核心要素4

每个工具独立的容错逻辑：
    SAM2      失败时退回 DINOv2 原始 mask
    VLM       幻觉过滤 + 格式校验
    DINOv2    wall 覆盖率合理性检查
    矢量化     wall_boxes 数量和尺寸合理性检查
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import logger
from contracts import ValidationError
from skeleon.tool_configs import SAM2_CFG, VLM_CFG


# ══════════════════════════════════════════════════════════════
# DINOv2 输出验证
# ══════════════════════════════════════════════════════════════

def validate_inference_output(
    wall_mask:  np.ndarray,
    det_boxes:  np.ndarray,
    image_wh:   Tuple[int, int],
    warnings:   List[str],
) -> bool:
    """
    检查 DINOv2 推理输出是否合理。
    返回 True=合格，False=需要重试或跳过。

    合理性标准：
        wall 覆盖率：5% ~ 60%（太低=没检测到墙，太高=全图都是墙）
        检测框：坐标在图像范围内，宽高 > 5px
    """
    W, H = image_wh
    ok   = True

    # 1. wall 覆盖率
    coverage = float(wall_mask.mean())
    if coverage < 0.05:
        warnings.append(
            f'[DINOv2] wall 覆盖率过低: {coverage*100:.1f}%  '
            f'（可能未检测到墙体，建议降低推理阈值）'
        )
        ok = False
    elif coverage > 0.60:
        warnings.append(
            f'[DINOv2] wall 覆盖率过高: {coverage*100:.1f}%  '
            f'（可能误检，建议提高推理阈值）'
        )
        # 过高不阻断，只警告

    # 2. 检测框范围检查
    n_invalid = 0
    for b in det_boxes:
        x1, y1, x2, y2 = [int(v) for v in b]
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            n_invalid += 1
        elif (x2 - x1) < 5 or (y2 - y1) < 5:
            n_invalid += 1
    if n_invalid > 0:
        warnings.append(f'[DINOv2] {n_invalid} 个检测框无效（越界或过小）')

    return ok


# ══════════════════════════════════════════════════════════════
# SAM2 容错
# ══════════════════════════════════════════════════════════════

def sam2_with_fallback(
    image_rgb:    np.ndarray,
    initial_mask: np.ndarray,
    predictor:    Any,
    warnings:     List[str],
) -> np.ndarray:
    """
    带容错的 SAM2 精化。

    失败策略（来自 SAM2_CFG.on_failure）：
        'skip'  → 返回 initial_mask，记录 Warning
        'retry' → 换一批随机点重试（最多 retry_times 次）

    SAM2 可能失败的场景：
        - 提示点全部落在背景区域（mask 太小，采不到前景点）
        - 图片分辨率太低
        - 显存临界，SAM2 内部 OOM
    """
    from tools.sam2_refine import refine_mask_with_sam2, sample_points_from_mask
    from config import CFG

    # 前置检查：mask 是否足够大，能采到正点
    pos_yx = np.argwhere(initial_mask > 0)
    if len(pos_yx) < SAM2_CFG.n_pos_points:
        warnings.append(
            f'[SAM2] 前景点不足（前景像素={len(pos_yx)}，'
            f'需要 {SAM2_CFG.n_pos_points} 个提示点），跳过精化'
        )
        return initial_mask

    last_exc = None
    for attempt in range(max(1, SAM2_CFG.retry_times if SAM2_CFG.on_failure == 'retry' else 1)):
        try:
            refined = refine_mask_with_sam2(image_rgb, initial_mask, predictor, CFG)

            # 后置检查：精化结果不能与原始 mask 偏差太大
            init_bool    = initial_mask.astype(bool)
            refined_bool = refined.astype(bool)
            inter = (init_bool & refined_bool).sum()
            union = (init_bool | refined_bool).sum()
            iou   = float(inter / union) if union > 0 else 0.0

            if iou < SAM2_CFG.min_iou_with_init:
                warnings.append(
                    f'[SAM2] 精化结果与原始 mask IoU 过低: {iou:.3f}'
                    f'（阈值 {SAM2_CFG.min_iou_with_init}），退回原始 mask'
                )
                return initial_mask

            logger.info(f'[SAM2] 精化成功  iou_with_init={iou:.3f}')
            return refined

        except Exception as e:
            last_exc = e
            warnings.append(f'[SAM2] 第{attempt+1}次尝试失败: {e}')
            if SAM2_CFG.on_failure != 'retry':
                break

    # 所有尝试失败
    warnings.append(
        f'[SAM2] 精化失败（{SAM2_CFG.on_failure}），退回 DINOv2 原始 mask。'
        f'最后错误: {last_exc}'
    )
    return initial_mask


# ══════════════════════════════════════════════════════════════
# VLM 输出验证 + 幻觉过滤
# ══════════════════════════════════════════════════════════════

def validate_vlm_output(
    raw_response:  str,
    image_wh:      Tuple[int, int],
    warnings:      List[str],
) -> Dict:
    """
    解析并校验 VLM 的 JSON 输出。

    两步校验：
        1. JSON 格式校验（解析失败返回空结果）
        2. 幻觉过滤（调用 VLM_CFG.filter_hallucinations）
    """
    # 1. 解析 JSON
    text = raw_response.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        end   = -1 if lines[-1].strip() == '```' else len(lines)
        text  = '\n'.join(lines[1:end])

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        warnings.append(f'[VLM] JSON 解析失败: {e}  原文前100字: {raw_response[:100]}')
        return {'openings': [], 'n_rooms': None, 'floor_area_m2': None, '_parse_error': True}

    # 2. 幻觉过滤
    openings      = parsed.get('openings', [])
    floor_area_m2 = parsed.get('floor_area_m2')
    filtered      = VLM_CFG.filter_hallucinations(
        openings      = openings,
        image_wh      = image_wh,
        floor_area_m2 = floor_area_m2,
        warnings      = warnings,
    )

    n_removed = len(openings) - len(filtered)
    if n_removed > 0:
        logger.info(f'[VLM] 幻觉过滤移除 {n_removed}/{len(openings)} 个开口')

    parsed['openings'] = filtered
    return parsed


# ══════════════════════════════════════════════════════════════
# 矢量化输出验证
# ══════════════════════════════════════════════════════════════

def validate_wall_boxes(
    wall_boxes: List,
    image_wh:   Tuple[int, int],
    warnings:   List[str],
) -> List:
    """
    过滤矢量化输出中的无效 wall_box。

    过滤规则：
        - bbox 在图像范围内
        - 面积 > 最小阈值（200 px²）
        - 宽高比合理（不超过 50:1，避免单像素线段）
    """
    W, H   = image_wh
    result = []
    MIN_AREA      = 200
    MAX_ASPECT    = 50.0

    for b in wall_boxes:
        x1, y1, x2, y2 = [int(v) for v in b]

        # 边界裁剪
        x1 = max(0, min(x1, W)); x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H)); y2 = max(0, min(y2, H))

        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            warnings.append(f'[矢量化] 跳过零尺寸 bbox: ({x1},{y1},{x2},{y2})')
            continue

        area   = bw * bh
        aspect = max(bw, bh) / max(min(bw, bh), 1)

        if area < MIN_AREA:
            warnings.append(f'[矢量化] 跳过过小 bbox: area={area}px²')
            continue

        if aspect > MAX_ASPECT:
            warnings.append(f'[矢量化] 跳过异常长宽比 bbox: {aspect:.1f}:1')
            continue

        result.append((x1, y1, x2, y2))

    n_removed = len(wall_boxes) - len(result)
    if n_removed > 0:
        logger.info(f'[矢量化] 过滤 {n_removed}/{len(wall_boxes)} 个无效 wall_box')

    if len(result) == 0:
        raise ValidationError(
            f'矢量化后 wall_boxes 为空（原始 {len(wall_boxes)} 个全被过滤），'
            f'图像尺寸={W}x{H}'
        )

    return result
