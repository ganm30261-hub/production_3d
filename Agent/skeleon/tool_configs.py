# skeleon/tool_configs.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

工具参数配置：集成期核心要素2

所有工具的超参数集中在这一个文件里。
想调某个工具的阈值，只改这里，不改算法代码。

五类配置：
    DINOv2Config     模型推理参数
    SAM2Config       SAM2 提示点策略
    VLMConfig        Prompt 模板 + 幻觉过滤规则
    VRAMConfig       显存分级预算
    DistortionConfig 畸变矫正阈值
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════
# 1. DINOv2 推理参数
# ══════════════════════════════════════════════════════════════

@dataclass
class DINOv2Config:
    """
    DINOv2+LoRA 滑动窗口推理参数。
    tile_size 必须是 14 的倍数（patch_size=14）。
    """
    # 权重路径（按优先级查找）
    ckpt_search_dirs: List[str] = field(default_factory=lambda: [
        '/workspace/production_3d/checkpoints_dinov2_lora/combined',
        '/workspace/production_3d/checkpoints_dinov2_lora/hq',
        '/workspace/production_3d/checkpoints_dinov2_lora/hq_arch',
    ])
    ckpt_pattern:     str   = '*_best.pth'

    # 推理参数
    tile_size:        int   = 518    # 37 × 14
    tile_overlap:     int   = 56     # 4 × 14，stride = 462 = 33 × 14
    wall_prob_thresh: float = 0.5    # wall mask 二值化阈值
    det_score_thresh: float = 0.5    # 检测框置信度过滤

    # 归一化（ImageNet）
    norm_mean:        Tuple = (0.485, 0.456, 0.406)
    norm_std:         Tuple = (0.229, 0.224, 0.225)

    # 显存预算（MB）
    vram_required_mb: int   = 6000   # RTX 5060 16GB，DINOv2-L 约 6GB

    def find_checkpoint(self) -> Optional[str]:
        """按优先级搜索最新的 best checkpoint。"""
        import glob
        for d in self.ckpt_search_dirs:
            matches = sorted(glob.glob(os.path.join(d, self.ckpt_pattern)))
            if matches:
                return matches[-1]
        return None


# ══════════════════════════════════════════════════════════════
# 2. SAM2 提示点策略
# ══════════════════════════════════════════════════════════════

@dataclass
class SAM2Config:
    """
    SAM2 提示点采样策略。

    提示点策略说明：
        正点：从 DINOv2 mask 腐蚀后的前景区域随机采样
              腐蚀目的：避免采到噪声边界附近，提高精度
        负点：从背景区域随机采样，帮助 SAM2 定位边界
        多候选：multimask_output=True 生成 3 个候选
              选与原始 mask IoU 最大的（语义一致性优先）
    """
    ckpt_path:         str   = '/workspace/checkpoints/sam2_hiera_large.pt'
    model_cfg:         str   = 'sam2_hiera_l.yaml'

    # 采样策略
    n_pos_points:      int   = 5     # 正点数量（前景）
    n_neg_points:      int   = 3     # 负点数量（背景）
    erode_px:          int   = 8     # 腐蚀半径（像素）

    # 质量过滤
    score_thresh:      float = 0.80  # SAM2 输出的最低置信度
    min_iou_with_init: float = 0.30  # 与初始 mask 的最低 IoU（防止语义漂移）

    # 失败处理策略
    # 'skip'    → 跳过精化，直接用 DINOv2 原始 mask
    # 'retry'   → 换一批随机点重试（最多 retry_times 次）
    on_failure:        str   = 'skip'
    retry_times:       int   = 2

    # 显存预算（MB）
    vram_required_mb:  int   = 4000  # SAM2-L 约 4GB


# ══════════════════════════════════════════════════════════════
# 3. VLM 参数 + Prompt 模板 + 幻觉过滤
# ══════════════════════════════════════════════════════════════

# 合理的门窗尺寸范围（米）
DOOR_WIDTH_RANGE   = (0.6, 2.4)   # 最窄单门 ~ 双开门
WINDOW_WIDTH_RANGE = (0.3, 4.0)   # 小气窗 ~ 落地窗
MAX_OPENINGS_PER_M2 = 0.5         # 每平方米最多 0.5 个开口（密度过高=幻觉）

# 合理的房间类型列表（出现范围外的类型视为幻觉）
VALID_ROOM_TYPES = {
    'living_room', 'bedroom', 'kitchen', 'bathroom', 'toilet',
    'dining_room', 'study', 'corridor', 'hallway', 'balcony',
    'storage', 'garage', 'laundry', 'entrance', 'unknown',
}

# 不合理的房间类型组合（同一楼层出现则报 Warning）
IMPOSSIBLE_COMBOS: List[Tuple[str, str]] = [
    ('swimming_pool', 'bedroom'),   # 游泳池不会出现在住宅平面图里
    ('gymnasium', 'toilet'),        # 住宅不会有体育馆
]


@dataclass
class VLMConfig:
    """
    VLM 语义补全参数。

    Prompt 设计原则：
        - 不传原始图片（太多 token），只传 mask 叠加可视化
        - 用结构化 JSON 约束输出格式，减少解析失败
        - Few-shot 示例内嵌在 system prompt 里（固定，不动态检索）
    """
    model:          str  = 'claude-sonnet-4-20250514'
    max_tokens:     int  = 1024
    temperature:    float = 0.0    # 确定性输出，减少幻觉

    # 显存预算（VLM 走 API，不占本地 GPU）
    vram_required_mb: int = 0

    # 幻觉过滤参数
    door_width_range:    Tuple = DOOR_WIDTH_RANGE
    window_width_range:  Tuple = WINDOW_WIDTH_RANGE
    max_openings_per_m2: float = MAX_OPENINGS_PER_M2
    valid_room_types:    set   = field(default_factory=lambda: VALID_ROOM_TYPES.copy())

    # System prompt（锁定格式）
    system_prompt: str = """\
你是一位专业的建筑平面图识别工程师。
你将收到一张平面图的墙体 mask 可视化图（绿色=墙体）。
请识别所有门和窗，输出严格的 JSON，不含 Markdown。

Few-shot 示例（标准输出格式）：
{
  "openings": [
    {"type":"door",   "bbox":[120,10,180,20], "wall_side":"north", "estimated_width_m":0.9,  "confidence":0.95},
    {"type":"window", "bbox":[200,10,260,20], "wall_side":"north", "estimated_width_m":1.2,  "confidence":0.88}
  ],
  "n_rooms": 3,
  "floor_area_m2": 85.0
}

规则：
  - bbox 格式: [x1, y1, x2, y2] 像素坐标
  - wall_side: north/south/east/west/unknown
  - estimated_width_m: 门 0.6~2.4，窗 0.3~4.0
  - confidence: 0.0~1.0\
"""

    def filter_hallucinations(
        self,
        openings: List[dict],
        image_wh: Tuple[int, int],
        floor_area_m2: Optional[float],
        warnings: List[str],
    ) -> List[dict]:
        """
        过滤 VLM 幻觉，返回合理的 openings 列表。
        不合理的项目记录到 warnings，不直接报错。
        """
        W, H   = image_wh
        result = []

        for op in openings:
            keep   = True
            reason = None

            # 1. 尺寸合理性
            w_m = op.get('estimated_width_m')
            if w_m is not None:
                if op['type'] == 'door' and not (
                    self.door_width_range[0] <= w_m <= self.door_width_range[1]
                ):
                    reason = f'门宽度不合理: {w_m}m（合理范围 {self.door_width_range}）'
                    keep   = False
                elif op['type'] == 'window' and not (
                    self.window_width_range[0] <= w_m <= self.window_width_range[1]
                ):
                    reason = f'窗宽度不合理: {w_m}m（合理范围 {self.window_width_range}）'
                    keep   = False

            # 2. bbox 在图像范围内
            if keep and 'bbox' in op:
                x1, y1, x2, y2 = op['bbox']
                if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                    reason = f'bbox 超出图像边界: {op["bbox"]} 图像={W}x{H}'
                    keep   = False
                elif x2 <= x1 or y2 <= y1:
                    reason = f'bbox 无效（宽或高为0）: {op["bbox"]}'
                    keep   = False

            # 3. 置信度过滤
            if keep and op.get('confidence', 1.0) < 0.5:
                reason = f'置信度过低: {op["confidence"]:.2f}'
                keep   = False

            if keep:
                result.append(op)
            else:
                warnings.append(f'[VLM幻觉过滤] 移除 {op.get("type","?")}  原因: {reason}')

        # 4. 开口密度检查（整体）
        if floor_area_m2 and floor_area_m2 > 0:
            density = len(result) / floor_area_m2
            if density > self.max_openings_per_m2:
                warnings.append(
                    f'[VLM幻觉过滤] 开口密度过高: {density:.2f}/m²'
                    f'（阈值 {self.max_openings_per_m2}），结果可能含幻觉'
                )

        return result


# ══════════════════════════════════════════════════════════════
# 4. 显存分级预算
# ══════════════════════════════════════════════════════════════

@dataclass
class VRAMConfig:
    """
    RTX 5060 显存调度策略。

    加载顺序（串行，不同时在 GPU 上）：
        DINOv2 → 卸载 → SAM2 → 卸载 → 训练（独占）

    每个模型的显存预算（MB）：
        DINOv2-L + LoRA  : ~6000 MB
        SAM2-Large       : ~4000 MB
        训练（梯度+优化器）: ~12000 MB（需要独占）
        VLM              : 0 MB（API 调用）
    """
    total_mb:          int = 16000   # RTX 5060 16GB

    # 各模型预算
    dinov2_mb:         int = 6000
    sam2_mb:           int = 4000
    training_mb:       int = 12000
    vlm_mb:            int = 0       # API，不占本地显存

    # 安全余量（低于此值时不加载新模型）
    safety_margin_mb:  int = 1024    # 1GB 安全余量

    # 模型切换策略
    # 'always_unload' : 每次工具调用前都卸载上一个模型（最安全，稍慢）
    # 'lazy_unload'   : 显存不足时才卸载（快，有碎片化风险）
    unload_strategy:   str = 'always_unload'

    @property
    def available_mb(self) -> int:
        """可用于加载模型的显存（总量 - 安全余量）。"""
        return self.total_mb - self.safety_margin_mb

    def can_fit(self, model_mb: int) -> bool:
        """判断某个模型能否加载（不考虑当前占用）。"""
        return model_mb <= self.available_mb


# ══════════════════════════════════════════════════════════════
# 5. 畸变矫正阈值
# ══════════════════════════════════════════════════════════════

@dataclass
class DistortionConfig:
    """
    畸变矫正阈值。
    判定"这张图太歪了"的数学标准。
    """
    # 旋转角度（度），超过此值认为需要矫正
    max_rotation_deg:    float = 5.0

    # 长宽比，超过此值认为图片被拉伸
    max_aspect_ratio:    float = 3.0
    min_aspect_ratio:    float = 0.3

    # Hough 直线检测：检测到的主要角度偏差超过此值触发矫正
    hough_angle_thresh:  float = 3.0    # 度

    # 失败处理：矫正失败时的策略
    # 'skip'     → 跳过矫正，直接用原图
    # 'reject'   → 拒绝该图，记录到错题集
    on_failure:          str   = 'skip'


# ══════════════════════════════════════════════════════════════
# 全局默认实例
# ══════════════════════════════════════════════════════════════

DINOV2_CFG    = DINOv2Config()
SAM2_CFG      = SAM2Config()
VLM_CFG       = VLMConfig()
VRAM_CFG      = VRAMConfig()
DISTORT_CFG   = DistortionConfig()
