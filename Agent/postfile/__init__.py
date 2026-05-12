# __init__.py
"""
后处理包（postprocess）

文件职责速查：
    postprocess_config.py  — 所有物理常数和算法阈值，想改参数只改这里
    vector_logic.py        — 算法层：mask → 矩形 wall_boxes（矢量化）
    reconstruct_3d.py      — 算法层：wall_boxes → 3D 网格 + 门窗布尔减法
    post_service.py        — 逻辑层：调度四步流水线，对外统一接口
    preview_service.py     — 服务层：生成预览并写入 staging 暂存区
    persistence_service.py — 服务层：审核结果处理（归档 / 拒绝 / 闭环）

典型调用链：
    preview_service.generate_preview()
        └→ post_service.run_postprocess()
               ├→ post_service.run_inference()        # 模型推理
               ├→ vector_logic.vectorize_wall_mask()  # 矢量化
               ├→ reconstruct_3d.match_openings_to_walls()
               └→ reconstruct_3d.build_3d_model()     # 3D 重建

审核链：
    persistence_service.approve()  # 通过 → archive
    persistence_service.reject()   # 拒绝 → 记录闭环原因
"""

# ── 配置层（最先 import，其他模块依赖它）──
from .postprocess_config import (
    VectorizationConfig,
    InferenceConfig,
    ValidationTargets,
    StagingConfig,
    VECT_CFG,
    INF_CFG,
    VAL_TARGETS,
    STAGING_CFG,
    WALL_HEIGHT,
    DOOR_HEIGHT,
    WINDOW_HEIGHT,
    WINDOW_SILL,
    PIXELS_PER_METER,
)

# ── 算法层 ──
from .vector_logic import (
    vectorize_wall_mask,
    compute_vectorization_iou,
    wall_boxes_to_mask,
)

from .reconstruct_3d import (
    match_openings_to_walls,
    build_3d_model,
)

# ── 逻辑层（懒加载模型，import 时不触发 GPU 初始化）──
from .post_service import (
    run_inference,
    run_postprocess,
    validate_single_sample,
    batch_validate,
    analyze_failures,
)

# ── 服务层 ──
from .preview_service import generate_preview

from .persistence_service import (
    approve,
    reject,
    list_pending,
    rejection_summary,
)

__all__ = [
    # 配置
    "VectorizationConfig",
    "InferenceConfig",
    "ValidationTargets",
    "StagingConfig",
    "VECT_CFG",
    "INF_CFG",
    "VAL_TARGETS",
    "STAGING_CFG",
    "WALL_HEIGHT",
    "DOOR_HEIGHT",
    "WINDOW_HEIGHT",
    "WINDOW_SILL",
    "PIXELS_PER_METER",
    # 算法
    "vectorize_wall_mask",
    "compute_vectorization_iou",
    "wall_boxes_to_mask",
    "match_openings_to_walls",
    "build_3d_model",
    # 流水线
    "run_inference",
    "run_postprocess",
    "validate_single_sample",
    "batch_validate",
    "analyze_failures",
    # 服务
    "generate_preview",
    "approve",
    "reject",
    "list_pending",
    "rejection_summary",
]
