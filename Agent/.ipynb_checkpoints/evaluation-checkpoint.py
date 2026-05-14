# evaluation.py
"""
综合评估模块：实现 Agent 的多维度决策准则

    S_total = w1 * IoU_pixel + w2 * C_topological + w3 * S_semantic

三个维度：
    IoU_pixel      : 像素级重合度（基础精度）
    C_topological  : 拓扑闭合性（物理逻辑，如墙体是否围成封闭房间）
    S_semantic     : 语义合理性（VLM 评分，如厨房和卫生间的空间逻辑）

各维度均归一化到 [0, 1]，权重之和应为 1.0。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import logger, PseudoLabelConfig


# ══════════════════════════════════════════════════════════════
# 评分权重配置
# ══════════════════════════════════════════════════════════════

@dataclass
class EvalWeights:
    """
    三维评分的权重配置。
    w1 + w2 + w3 应等于 1.0，否则 S_total 不在 [0,1] 内。
    S_semantic 调用 VLM，成本较高；无 VLM 时自动退化为两维评分。
    """
    w1_iou:        float = 0.50   # 像素级 IoU
    w2_topology:   float = 0.30   # 拓扑闭合性
    w3_semantic:   float = 0.20   # VLM 语义合理性

    def renormalize(self) -> EvalWeights:
        """权重归一化（当 semantic 不可用时按比例重新分配）。"""
        total = self.w1_iou + self.w2_topology + self.w3_semantic
        return EvalWeights(
            w1_iou      = self.w1_iou      / total,
            w2_topology = self.w2_topology / total,
            w3_semantic = self.w3_semantic / total,
        )


@dataclass
class EvalResult:
    """单张图的综合评估结果，写入推理日志和错题集。"""
    iou_pixel:    float                    # 维度1
    c_topological: float                   # 维度2
    s_semantic:   float                    # 维度3（无 VLM 时为 -1.0）
    s_total:      float                    # 加权总分
    weights:      EvalWeights
    details:      dict = field(default_factory=dict)  # 各维度的诊断细节

    @property
    def passed(self) -> bool:
        """是否达到可接受质量阈值（总分 >= 0.75）。"""
        return self.s_total >= 0.75

    def to_dict(self) -> dict:
        return {
            "iou_pixel":     round(self.iou_pixel,     4),
            "c_topological": round(self.c_topological, 4),
            "s_semantic":    round(self.s_semantic,     4),
            "s_total":       round(self.s_total,        4),
            "passed":        self.passed,
            "weights":       {
                "w1_iou":      self.weights.w1_iou,
                "w2_topology": self.weights.w2_topology,
                "w3_semantic": self.weights.w3_semantic,
            },
            "details": self.details,
        }


# ══════════════════════════════════════════════════════════════
# 维度1：像素级 IoU
# ══════════════════════════════════════════════════════════════

def compute_iou_pixel(
    pred_mask: np.ndarray,
    gt_mask:   np.ndarray,
) -> Tuple[float, dict]:
    """
    计算 wall 类别（前景=1）的像素级 IoU。

    当没有 GT 标注时（gt_mask 全零），返回 IoU=1.0 并在 details 里注明，
    避免第一步推理因为没有参照而被错误惩罚。

    返回:
        iou     : float [0, 1]
        details : dict  诊断信息
    """
    pred = pred_mask.astype(bool).ravel()
    gt   = gt_mask.astype(bool).ravel()

    if gt.sum() == 0:
        return 1.0, {"note": "no_gt_mask，跳过 IoU 计算"}

    tp    = (pred & gt).sum()
    fp    = (pred & ~gt).sum()
    fn    = (~pred & gt).sum()
    denom = tp + fp + fn

    iou = float(tp / denom) if denom > 0 else 0.0
    return iou, {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "pred_positive": int(pred.sum()),
        "gt_positive":   int(gt.sum()),
    }


# ══════════════════════════════════════════════════════════════
# 维度2：拓扑闭合性
# ══════════════════════════════════════════════════════════════

def compute_topological_closure(
    wall_boxes:  List[Tuple],
    image_wh:    Tuple[int, int],
    min_room_area: int = 2000,
) -> Tuple[float, dict]:
    """
    评估墙体的拓扑闭合性：墙体是否围成有效的封闭房间。

    算法：
        1. 把 wall_boxes 光栅化成 binary mask
        2. 对前景取反（背景即"室内空间"）
        3. 用 connectedComponentsWithStats 找连通区域
        4. 过滤掉外部大背景和过小噪声，剩下的算作"有效房间"
        5. 闭合率 = 有效房间数 / max(1, 理论最小房间数)
           归一化到 [0, 1]

    wall_boxes : List[Tuple[x1,y1,x2,y2]]
    image_wh   : (W, H) 原图尺寸
    min_room_area : 低于此像素面积的连通区不算有效房间（噪声过滤）

    返回:
        score   : float [0, 1]
        details : dict  诊断信息
    """
    W, H = image_wh

    if not wall_boxes:
        return 0.0, {"note": "wall_boxes 为空，无法评估拓扑"}

    # ── 光栅化 wall_boxes ──
    canvas = np.zeros((H, W), dtype=np.uint8)
    for b in wall_boxes:
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), 1, thickness=-1)

    # ── 找背景连通域（前景取反）──
    bg          = (canvas == 0).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)

    # label 0 是整张图的背景（最大连通域），跳过
    image_area  = H * W
    valid_rooms = []
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        # 过滤：太大（外部大背景）或太小（噪声）
        if min_room_area <= area <= image_area * 0.8:
            valid_rooms.append(area)

    n_rooms = len(valid_rooms)

    # 闭合率：有有效房间就给分，房间越多分越高，上限 1.0
    # 用 sigmoid-like 映射：1 个房间 ≈ 0.6，3 个 ≈ 0.88，5 个 ≈ 0.95
    if n_rooms == 0:
        score = 0.0
    else:
        score = float(1.0 - 1.0 / (1.0 + n_rooms * 0.8))

    return score, {
        "n_valid_rooms":   n_rooms,
        "room_areas":      valid_rooms[:10],   # 最多记录10个避免日志过大
        "n_wall_boxes":    len(wall_boxes),
        "wall_coverage_%": round(canvas.sum() / image_area * 100, 2),
    }


# ══════════════════════════════════════════════════════════════
# 维度3：VLM 语义合理性
# ══════════════════════════════════════════════════════════════

_SEMANTIC_SYSTEM = """\
你是一位建筑平面图质量审核专家。
你将收到一张平面图和对应的伪标注 JSON（包含墙体框、门窗位置、房间数量）。
请评估这份标注的语义合理性，重点检查：
  1. 门窗是否合理分布在墙体上（不应悬空）
  2. 空间逻辑是否自洽（房间数量与面积比例是否正常）
  3. 是否存在明显的语义矛盾（如极小房间里塞满开口）
输出必须是合法 JSON，不含 Markdown。\
"""

_SEMANTIC_USER = """\
伪标注摘要：
{annotation_summary}

请输出：
{{
  "score": 0.0到1.0之间的浮点数（1.0=完全合理，0.0=严重错误）,
  "issues": ["问题描述1", "问题描述2"],   // 空列表表示无问题
  "suggestions": ["改进建议1"]            // 可选
}}
"""


def compute_semantic_score(
    image_rgb:   np.ndarray,
    wall_boxes:  List[Tuple],
    openings:    List[dict],
    cfg:         PseudoLabelConfig,
    client,
) -> Tuple[float, dict]:
    """
    调用 VLM 对标注结果做语义合理性评分。

    为节省 token，不传图片，只传结构化的标注摘要（文本）。
    如果 client 为 None，返回 score=-1.0 表示"未评估"。

    返回:
        score   : float [0, 1]，或 -1.0 表示跳过
        details : dict
    """
    if client is None:
        return -1.0, {"note": "vlm_client=None，跳过语义评分"}

    try:
        import anthropic as _anthropic
    except ImportError:
        return -1.0, {"note": "anthropic 未安装，跳过语义评分"}

    # 构造轻量文本摘要（不传图片，节省 token）
    n_doors   = sum(1 for o in openings if o.get("type") == "door")
    n_windows = sum(1 for o in openings if o.get("type") == "window")
    h, w      = image_rgb.shape[:2]

    summary = json.dumps({
        "image_size":    f"{w}x{h}",
        "n_wall_boxes":  len(wall_boxes),
        "n_doors":       n_doors,
        "n_windows":     n_windows,
        "wall_density_%": round(len(wall_boxes) / max(w * h / 10000, 1), 2),
        "openings_sample": openings[:5],   # 只传前5个，控制 token
    }, ensure_ascii=False, indent=2)

    try:
        response = client.messages.create(
            model      = cfg.vlm_model,
            max_tokens = 512,
            system     = _SEMANTIC_SYSTEM,
            messages   = [{
                "role": "user",
                "content": _SEMANTIC_USER.format(annotation_summary=summary),
            }],
        )
        text = response.content[0].text.strip()
        # 去掉可能的 ```json 包裹
        if text.startswith("```"):
            lines = text.split("\n")
            text  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        parsed    = json.loads(text)
        score     = float(parsed.get("score", 0.5))
        score     = max(0.0, min(1.0, score))   # clamp
        details   = {
            "issues":      parsed.get("issues", []),
            "suggestions": parsed.get("suggestions", []),
            "tokens": {
                "input":  response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        }
        return score, details

    except Exception as e:
        logger.warning(f"语义评分失败: {e}")
        return 0.5, {"note": f"VLM 调用异常，退回默认分 0.5: {e}"}


# ══════════════════════════════════════════════════════════════
# 综合评分入口
# ══════════════════════════════════════════════════════════════

def evaluate(
    pred_mask:  np.ndarray,
    gt_mask:    np.ndarray,
    wall_boxes: List[Tuple],
    openings:   List[dict],
    image_rgb:  np.ndarray,
    cfg:        PseudoLabelConfig,
    weights:    Optional[EvalWeights] = None,
    vlm_client  = None,
) -> EvalResult:
    """
    计算综合评分 S_total = w1*IoU + w2*C_topo + w3*S_semantic。

    参数:
        pred_mask  : (H,W) uint8  当前预测的 wall mask
        gt_mask    : (H,W) uint8  GT mask（无标注时传全零）
        wall_boxes : Shrinking 矢量化输出
        openings   : VLM 语义补全输出
        image_rgb  : (H,W,3) uint8 原图（语义评分用）
        cfg        : PseudoLabelConfig
        weights    : EvalWeights（None 时用默认权重）
        vlm_client : anthropic.Anthropic()（None 时跳过语义评分）

    返回:
        EvalResult  包含三个维度分数、总分、诊断细节
    """
    W      = EvalWeights() if weights is None else weights
    H, Wd  = pred_mask.shape
    image_wh = (Wd, H)

    # ── 维度1：像素 IoU ──
    iou, iou_details = compute_iou_pixel(pred_mask, gt_mask)

    # ── 维度2：拓扑闭合 ──
    c_topo, topo_details = compute_topological_closure(wall_boxes, image_wh)

    # ── 维度3：VLM 语义（可选）──
    s_sem, sem_details = compute_semantic_score(
        image_rgb, wall_boxes, openings, cfg, vlm_client
    )

    # ── 权重归一化（语义不可用时重分配）──
    if s_sem < 0:
        w_eff = EvalWeights(
            w1_iou      = W.w1_iou,
            w2_topology = W.w2_topology,
            w3_semantic = 0.0,
        ).renormalize()
        s_sem_eff = 0.0
    else:
        w_eff     = W
        s_sem_eff = s_sem

    s_total = (
        w_eff.w1_iou      * iou    +
        w_eff.w2_topology * c_topo +
        w_eff.w3_semantic * s_sem_eff
    )

    result = EvalResult(
        iou_pixel     = iou,
        c_topological = c_topo,
        s_semantic    = s_sem,
        s_total       = round(s_total, 4),
        weights       = w_eff,
        details       = {
            "iou":      iou_details,
            "topology": topo_details,
            "semantic": sem_details,
        },
    )

    logger.info(
        f"评估完成  "
        f"IoU={iou:.3f}  "
        f"Topo={c_topo:.3f}  "
        f"Sem={s_sem:.3f}  "
        f"Total={s_total:.3f}  "
        f"{'✓ passed' if result.passed else '✗ failed'}"
    )
    return result
