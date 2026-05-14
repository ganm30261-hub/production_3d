"""
postprocess_config.py — 后处理专用配置层
所有矢量化和3D重建的物理常数、算法阈值集中在这里。

规则：想调整"墙有多厚"或"门的高度"，只改这一个文件。
      训练超参数在 config.py，后处理参数在这里，互不影响。
"""

from dataclasses import dataclass
import os


# ══════════════════════════════════════════════════════════════
# 路径配置（输出目录）
# ══════════════════════════════════════════════════════════════

def _detect_output_dir() -> str:
    if os.path.exists('/workspace/production_3d'):
        return '/workspace/production_3d/outputs'
    elif os.path.exists('/content'):
        return '/content/outputs'
    return './outputs'


OUTPUT_DIR      = _detect_output_dir()
STAGING_DIR     = os.path.join(OUTPUT_DIR, 'staging')    # 待审核区
ARCHIVE_DIR     = os.path.join(OUTPUT_DIR, 'archive')    # 已归档区
TEMP_DIR        = os.path.join(OUTPUT_DIR, 'temp')       # 临时文件


# ══════════════════════════════════════════════════════════════
# 物理常数（Section 2.5）
# ══════════════════════════════════════════════════════════════

WALL_HEIGHT      = 2.8    # 米，标准层高
DOOR_HEIGHT      = 2.1    # 米，门洞高度
WINDOW_HEIGHT    = 1.2    # 米，窗洞高度
WINDOW_SILL      = 0.9    # 米，窗台距地高度
PIXELS_PER_METER = 50     # 像素/米（根据实际图纸比例调整）

FLOOR_THICKNESS  = 0.05   # 米，地板厚度（3D 模型）


# ══════════════════════════════════════════════════════════════
# 矢量化算法阈值（Section 2.4）
# ══════════════════════════════════════════════════════════════

@dataclass
class VectorizationConfig:
    """
    Section 2.4 墙体矢量化配置
    每个字段都有注释说明"调大/调小会发生什么"
    """

    # Shrinking 算法目标 IoU（越高矩形越贴合轮廓，计算越慢）
    iou_threshold: float = 0.85

    # 墙体轮廓最小面积（像素^2），过小的轮廓视为噪声
    min_segment_area: int = 200

    # 重叠处理：两个 bbox 的 overlap / 较小bbox面积 > 此阈值才做裁剪
    overlap_thresh: float = 0.1

    # Shrinking 算法：bbox 的最小边长（像素），小于此值停止收缩
    shrink_min_size: int = 5

    # Hough 变换：角度分辨率（度），越小越精细但越慢
    hough_angle_resolution: float = 1.0

    # Hough 变换：最多保留的主要角度数
    hough_n_angles: int = 4

    # 形态学 opening kernel 尺寸（去噪）
    morph_open_size: int = 3

    # 形态学 closing kernel 尺寸（填洞）
    morph_close_size: int = 5

    # 提取水平/垂直段的 kernel 长度（越长要求线段越长才能通过）
    morph_horizontal_len: int = 20
    morph_vertical_len:   int = 20


# ══════════════════════════════════════════════════════════════
# 推理配置（后处理用）
# ══════════════════════════════════════════════════════════════

@dataclass
class InferenceConfig:
    """
    模型推理时的参数（后处理流程调用推理时使用）
    """
    tile_size:        int   = 512
    tile_overlap:     int   = 64
    det_score_thresh: float = 0.5    # 检测置信度过滤阈值
    nms_iou_thresh:   float = 0.5    # NMS 阈值


# ══════════════════════════════════════════════════════════════
# 验证指标目标值（对应论文 Table 1）
# ══════════════════════════════════════════════════════════════

@dataclass
class ValidationTargets:
    """
    后处理验证系统的通过/不通过判断阈值
    来源：论文 Table 1 + 工程经验
    """
    iou_mask_min:    float = 0.75    # 分割 IoU 目标（论文 0.81）
    iou_vect_min:    float = 0.60    # 矢量化 IoU 目标（论文 0.80）
    det_f1_min:      float = 0.70    # 门窗检测 F1 目标
    match_rate_min:  float = 0.70    # 门窗落在墙上的比例
    geom_ok_min:     float = 0.90    # 3D 几何合理性（bbox 宽高比）
    det_score_thresh: float = 0.5    # 验证时的检测过滤阈值
    det_iou_thresh:   float = 0.5    # 验证时的 IoU 匹配阈值
    match_overlap_thresh: float = 0.3  # 门窗落墙判断的重叠比例


# ══════════════════════════════════════════════════════════════
# 人工审核流程配置（Staging / Archive）
# ══════════════════════════════════════════════════════════════

@dataclass
class StagingConfig:
    """
    人工审核流水线配置
    CPU Worker 生成 .glb → staging；审核通过 → archive；
    审核不通过 → 记录原因（数据闭环）
    """
    staging_dir:    str = STAGING_DIR
    archive_dir:    str = ARCHIVE_DIR
    temp_dir:       str = TEMP_DIR

    # GCS 配置（如果启用远程归档）
    gcs_bucket:     str = 'yalingdata'
    gcs_staging:    str = 'staging/floorplan_3d'
    gcs_archive:    str = 'archive/floorplan_3d'

    # 预签名 URL 过期时间（秒），用于前端预览
    preview_url_ttl: int = 86400   # 24 小时

    # 不合格原因枚举（用于数据闭环标注）
    REJECTION_REASONS = [
        'wall_missing',       # 墙体缺失
        'wall_shape_wrong',   # 墙体形状错误
        'door_offset',        # 门位置偏移
        'window_offset',      # 窗位置偏移
        'scale_wrong',        # 比例不对
        'geometry_invalid',   # 3D 几何错误
        'other',              # 其他
    ]


# ── 默认实例（直接 import 使用）──
VECT_CFG    = VectorizationConfig()
INF_CFG     = InferenceConfig()
VAL_TARGETS = ValidationTargets()
STAGING_CFG = StagingConfig()
