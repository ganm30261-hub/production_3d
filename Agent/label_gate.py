# label_gate.py
"""
支柱1：伪标签质量关卡

三道关卡决定一张伪标注的命运：
    PASS        → 进入黄金数据集，参与训练
    REVIEW      → 语义存疑，存入人工审核池
    REJECT      → 质量太差，直接丢弃

判断标准：
    关卡1  几何关卡   wall 覆盖率 + 房间闭合性（OpenCV）
    关卡2  模型关卡   SAM2 stability score + S_total 综合评分
    关卡3  语义关卡   VLM 通过率 + 没有门的房间检测
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import logger


# ══════════════════════════════════════════════════════════════
# 判定结果
# ══════════════════════════════════════════════════════════════

class GateResult(str, Enum):
    PASS   = "pass"    # 进入黄金数据集
    REVIEW = "review"  # 人工审核池
    REJECT = "reject"  # 直接丢弃


@dataclass
class GateDecision:
    result:       GateResult
    score:        float              # 综合质量分 0~1
    failed_gates: List[str]          # 哪些关卡不通过
    warnings:     List[str]
    details:      dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.result == GateResult.PASS

    def to_dict(self) -> dict:
        return {
            'result':       self.result.value,
            'score':        round(self.score, 4),
            'failed_gates': self.failed_gates,
            'warnings':     self.warnings,
            'details':      self.details,
        }


# ══════════════════════════════════════════════════════════════
# 阈值配置
# ══════════════════════════════════════════════════════════════

@dataclass
class GateConfig:
    # 关卡1：几何
    min_wall_coverage:   float = 0.05   # wall 覆盖率下限
    max_wall_coverage:   float = 0.60   # wall 覆盖率上限
    min_closed_rooms:    int   = 1      # 最少封闭房间数
    min_wall_boxes:      int   = 3      # 最少 wall_box 数量

    # 关卡2：模型质量
    min_sam2_stability:  float = 0.85   # SAM2 stability score 下限
    min_s_total_pass:    float = 0.60   # S_total >= 此值 → PASS
    min_s_total_review:  float = 0.40   # S_total >= 此值 → REVIEW，否则 REJECT

    # 关卡3：语义
    min_vlm_confidence:  float = 0.70   # VLM 平均置信度下限
    require_door:        bool  = False  # 是否强制要求有门（小户型可能没门）
    max_openings_per_wall: float = 3.0  # 每段墙最多几个开口

    # 输出路径
    golden_pool_path:   str = '/workspace/production_3d/outputs/golden_pool'
    review_pool_path:   str = '/workspace/production_3d/outputs/review_pool'
    reject_pool_path:   str = '/workspace/production_3d/outputs/reject_pool'


DEFAULT_GATE_CFG = GateConfig()


# ══════════════════════════════════════════════════════════════
# 关卡1：几何检查
# ══════════════════════════════════════════════════════════════

def check_geometry(
    wall_mask:  np.ndarray,
    wall_boxes: list,
    image_wh:   Tuple[int, int],
    cfg:        GateConfig,
) -> Tuple[bool, List[str], dict]:
    """
    几何关卡：检查墙体覆盖率和房间闭合性。
    返回 (通过, warnings, details)
    """
    W, H     = image_wh
    warnings = []
    details  = {}
    passed   = True

    # 1. wall 覆盖率
    coverage = float(wall_mask.mean())
    details['wall_coverage'] = round(coverage, 4)

    if coverage < cfg.min_wall_coverage:
        warnings.append(f'wall 覆盖率过低: {coverage*100:.1f}% < {cfg.min_wall_coverage*100:.0f}%')
        passed = False
    elif coverage > cfg.max_wall_coverage:
        warnings.append(f'wall 覆盖率过高: {coverage*100:.1f}% > {cfg.max_wall_coverage*100:.0f}%')
        passed = False

    # 2. wall_box 数量
    details['n_wall_boxes'] = len(wall_boxes)
    if len(wall_boxes) < cfg.min_wall_boxes:
        warnings.append(f'wall_boxes 数量不足: {len(wall_boxes)} < {cfg.min_wall_boxes}')
        passed = False

    # 3. 房间闭合性（背景连通域分析）
    canvas = np.zeros((H, W), dtype=np.uint8)
    for b in wall_boxes:
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), 1, -1)

    bg      = (canvas == 0).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)

    image_area   = H * W
    closed_rooms = []
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if 2000 <= area <= image_area * 0.8:
            closed_rooms.append(area)

    details['n_closed_rooms'] = len(closed_rooms)
    details['room_areas']     = closed_rooms[:5]

    if len(closed_rooms) < cfg.min_closed_rooms:
        warnings.append(
            f'封闭房间数不足: {len(closed_rooms)} < {cfg.min_closed_rooms}，墙体可能断裂'
        )
        passed = False

    return passed, warnings, details


# ══════════════════════════════════════════════════════════════
# 关卡2：模型质量检查
# ══════════════════════════════════════════════════════════════

def check_model_quality(
    s_total:        float,
    sam2_stability: Optional[float],
    cfg:            GateConfig,
) -> Tuple[GateResult, List[str], dict]:
    """
    模型质量关卡：根据综合评分决定 PASS / REVIEW / REJECT。
    返回 (GateResult, warnings, details)
    """
    warnings = []
    details  = {
        's_total':        round(s_total, 4),
        'sam2_stability': round(sam2_stability, 4) if sam2_stability else None,
    }

    # SAM2 stability score 检查
    if sam2_stability is not None and sam2_stability < cfg.min_sam2_stability:
        warnings.append(
            f'SAM2 stability score 过低: {sam2_stability:.3f} < {cfg.min_sam2_stability}'
        )

    # S_total 决策
    if s_total >= cfg.min_s_total_pass:
        result = GateResult.PASS
    elif s_total >= cfg.min_s_total_review:
        result = GateResult.REVIEW
        warnings.append(f'S_total={s_total:.3f} 低于 PASS 阈值，送审核池')
    else:
        result = GateResult.REJECT
        warnings.append(f'S_total={s_total:.3f} 过低，直接丢弃')

    # SAM2 不稳定时降级
    if sam2_stability is not None and sam2_stability < cfg.min_sam2_stability:
        if result == GateResult.PASS:
            result = GateResult.REVIEW
            warnings.append('因 SAM2 不稳定，从 PASS 降级为 REVIEW')

    return result, warnings, details


# ══════════════════════════════════════════════════════════════
# 关卡3：语义检查
# ══════════════════════════════════════════════════════════════

def check_semantics(
    openings:   list,
    wall_boxes: list,
    n_rooms:    Optional[int],
    cfg:        GateConfig,
) -> Tuple[bool, List[str], dict]:
    """
    语义关卡：检查门窗分布合理性。
    返回 (通过, warnings, details)
    """
    warnings = []
    details  = {}
    passed   = True

    doors   = [o for o in openings if o.get('type') == 'door']
    windows = [o for o in openings if o.get('type') == 'window']

    details['n_doors']   = len(doors)
    details['n_windows'] = len(windows)
    details['n_rooms']   = n_rooms

    # 1. 没有门的房间检查
    if cfg.require_door and len(doors) == 0 and (n_rooms or 0) > 0:
        warnings.append(f'存在 {n_rooms} 个房间但没有检测到门，语义可疑')
        passed = False

    # 2. 置信度检查
    confidences = [o.get('confidence', 1.0) for o in openings]
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        details['avg_opening_confidence'] = round(avg_conf, 4)
        if avg_conf < cfg.min_vlm_confidence:
            warnings.append(f'门窗平均置信度过低: {avg_conf:.3f} < {cfg.min_vlm_confidence}')
            passed = False

    # 3. 开口密度检查
    if wall_boxes and openings:
        ratio = len(openings) / len(wall_boxes)
        details['openings_per_wall'] = round(ratio, 2)
        if ratio > cfg.max_openings_per_wall:
            warnings.append(
                f'开口/墙体比例过高: {ratio:.1f} > {cfg.max_openings_per_wall}，可能存在幻觉'
            )
            passed = False

    return passed, warnings, details


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def run_gate(
    wall_mask:      np.ndarray,
    wall_boxes:     list,
    openings:       list,
    image_wh:       Tuple[int, int],
    s_total:        float,
    sam2_stability: Optional[float] = None,
    n_rooms:        Optional[int]   = None,
    cfg:            GateConfig      = DEFAULT_GATE_CFG,
) -> GateDecision:
    """
    运行三道质量关卡，返回 GateDecision。

    调用时机：pipeline 生成伪标注后，存入训练集之前。
    """
    all_warnings   = []
    all_details    = {}
    failed_gates   = []

    # 关卡1：几何
    geom_ok, geom_warn, geom_det = check_geometry(wall_mask, wall_boxes, image_wh, cfg)
    all_warnings.extend(geom_warn)
    all_details['geometry'] = geom_det
    if not geom_ok:
        failed_gates.append('geometry')

    # 关卡2：模型质量
    model_result, model_warn, model_det = check_model_quality(s_total, sam2_stability, cfg)
    all_warnings.extend(model_warn)
    all_details['model'] = model_det
    if model_result != GateResult.PASS:
        failed_gates.append('model_quality')

    # 关卡3：语义
    sem_ok, sem_warn, sem_det = check_semantics(openings, wall_boxes, n_rooms, cfg)
    all_warnings.extend(sem_warn)
    all_details['semantics'] = sem_det
    if not sem_ok:
        failed_gates.append('semantics')

    # 综合决策
    if model_result == GateResult.REJECT:
        final = GateResult.REJECT
    elif not geom_ok:
        final = GateResult.REJECT
    elif model_result == GateResult.REVIEW or not sem_ok:
        final = GateResult.REVIEW
    else:
        final = GateResult.PASS

    decision = GateDecision(
        result       = final,
        score        = s_total,
        failed_gates = failed_gates,
        warnings     = all_warnings,
        details      = all_details,
    )

    symbol = {'pass': '✓', 'review': '?', 'reject': '✗'}[final.value]
    logger.info(
        f'[LabelGate] {symbol} {final.value.upper()}  '
        f'S_total={s_total:.3f}  '
        f'failed={failed_gates}'
    )
    return decision


# ══════════════════════════════════════════════════════════════
# 分流存储
# ══════════════════════════════════════════════════════════════

def route_label(
    decision:   GateDecision,
    trace_id:   str,
    meta:       dict,
    cfg:        GateConfig = DEFAULT_GATE_CFG,
) -> str:
    """
    根据 GateDecision 把标注元数据写入对应的池子。
    返回写入的文件路径。
    """
    pool_map = {
        GateResult.PASS:   cfg.golden_pool_path,
        GateResult.REVIEW: cfg.review_pool_path,
        GateResult.REJECT: cfg.reject_pool_path,
    }
    pool_dir = pool_map[decision.result]
    os.makedirs(pool_dir, exist_ok=True)

    record = {
        'trace_id':  trace_id,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'gate':      decision.to_dict(),
        'meta':      meta,
    }
    path = os.path.join(pool_dir, f'{trace_id}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return path


def count_golden_pool(cfg: GateConfig = DEFAULT_GATE_CFG) -> int:
    """返回黄金数据集当前的样本数量（触发训练的判断依据）。"""
    if not os.path.exists(cfg.golden_pool_path):
        return 0
    return len([f for f in os.listdir(cfg.golden_pool_path) if f.endswith('.json')])
