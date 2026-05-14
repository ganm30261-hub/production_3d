# skeleon/integration.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

集成层：替换 orchestrator.py 里的 mock 函数

把骨架的三个 mock 函数替换成真实工具调用：
    _mock_run_labeling()      → run_real_labeling()
    _mock_run_training()      → run_real_training()
    _mock_run_reconstruction() → run_real_reconstruction()

每个函数都接入：
    VRAMScheduler  显存调度
    tool_validator 容错校验
    coord_utils    坐标系对齐
    contracts      格式验证
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import CFG, logger, PseudoLabelConfig
from skeleon.contracts import (
    LabelContract, WallBox, Opening, BBox,
    ReconstructContract, validate_handoff,
)
from skeleon.global_state import PipelineState
from skeleon.observability import TraceLogger
from skeleon.vram_scheduler import SCHEDULER, flush_vram
from skeleon.tool_validator import (
    validate_inference_output,
    sam2_with_fallback,
    validate_vlm_output,
    validate_wall_boxes,
)
from skeleon.tool_configs import SAM2_CFG
from utils.coord_utils import bbox_to_norm, clip_bbox


# ══════════════════════════════════════════════════════════════
# 真实标注流程（替换 _mock_run_labeling）
# ══════════════════════════════════════════════════════════════

def run_real_labeling(
    image_path: str,
    cfg:        PseudoLabelConfig,
    tl:         TraceLogger,
    warnings:   List[str],
) -> LabelContract:
    """
    真实标注流程：
        1. DINOv2+LoRA 推理      → wall_mask + det_boxes
        2. SAM2 精化（带容错）    → refined_mask
        3. 矢量化                 → wall_boxes
        4. VLM 语义补全（带幻觉过滤）→ openings
        5. 组装 LabelContract
    """
    # ── 读图 ──
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f'无法读取图片: {image_path}')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W    = img_rgb.shape[:2]
    tl.info(f'图片尺寸: {W}x{H}')

    # ════════════════════════════
    # Step 1: DINOv2 推理
    # ════════════════════════════
    tl.info('Step1: DINOv2+LoRA 推理')
    t0 = time.time()

    from pipeline import run_inference
    with SCHEDULER.load('dinov2') as model:
        infer_out = run_inference(img_rgb, model, cfg)

    wall_mask  = infer_out['wall_mask']
    det_boxes  = infer_out['boxes']
    det_labels = infer_out['labels']

    # 验证推理输出
    ok = validate_inference_output(wall_mask, det_boxes, (W, H), warnings)
    if not ok:
        tl.warning(f'DINOv2 输出质量不佳，wall%={wall_mask.mean()*100:.1f}%')

    tl.metric('wall_coverage', float(wall_mask.mean()), stage='labeling')
    tl.metric('det_boxes',     len(det_boxes),          stage='labeling')
    tl.info(f'Step1 完成  {time.time()-t0:.1f}s')

    # ════════════════════════════
    # Step 2: SAM2 精化
    # ════════════════════════════
    tl.info('Step2: SAM2 精化')
    t0 = time.time()

    if os.path.exists(SAM2_CFG.ckpt_path):
        with SCHEDULER.load('sam2') as predictor:
            refined_mask = sam2_with_fallback(img_rgb, wall_mask, predictor, warnings)
    else:
        warnings.append('[SAM2] 权重不存在，跳过精化，使用 DINOv2 原始 mask')
        refined_mask = wall_mask

    tl.info(f'Step2 完成  {time.time()-t0:.1f}s  refined_px={refined_mask.sum()}')
    flush_vram()

    # ════════════════════════════
    # Step 3: 矢量化
    # ════════════════════════════
    tl.info('Step3: 矢量化')
    t0 = time.time()

    try:
        from vector_logic import vectorize_wall_mask
        from postprocess_config import VectorizationConfig
        vect_cfg   = VectorizationConfig(iou_threshold=cfg.shrink_iou_thresh)
        raw_boxes  = vectorize_wall_mask(refined_mask, vect_cfg)
        wall_boxes = validate_wall_boxes(raw_boxes, (W, H), warnings)
    except ImportError:
        warnings.append('[矢量化] vector_logic 不可用，用检测框代替 wall_boxes')
        wall_boxes = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in det_boxes]

    tl.metric('n_wall_boxes', len(wall_boxes), stage='labeling')
    tl.info(f'Step3 完成  {time.time()-t0:.1f}s  wall_boxes={len(wall_boxes)}')

    # ════════════════════════════
    # Step 4: VLM 语义补全
    # ════════════════════════════
    tl.info('Step4: VLM 语义补全')
    t0 = time.time()

    try:
        import anthropic
        from tools.vlm_completion import vlm_semantic_completion

        client   = anthropic.Anthropic()
        response = client.messages.create(
            model      = cfg.vlm_model,
            max_tokens = cfg.vlm_max_tokens,
            system     = VLM_CFG.system_prompt,
            messages   = [{
                'role': 'user',
                'content': [
                    _make_mask_image_content(img_rgb, refined_mask),
                    {'type': 'text', 'text': '请识别所有门和窗，输出 JSON。'},
                ],
            }],
        )
        vlm_result = validate_vlm_output(
            response.content[0].text, (W, H), warnings
        )
    except Exception as e:
        warnings.append(f'[VLM] 调用失败，用检测框代替: {e}')
        vlm_result = _det_boxes_to_openings(det_boxes, det_labels)

    raw_openings = vlm_result.get('openings', [])
    tl.metric('n_openings', len(raw_openings), stage='labeling')
    tl.info(f'Step4 完成  {time.time()-t0:.1f}s  openings={len(raw_openings)}')

    # ════════════════════════════
    # Step 5: 组装 LabelContract
    # ════════════════════════════
    contract = LabelContract(
        image_path = image_path,
        image_wh   = (W, H),
        wall_boxes = [
            WallBox(bbox=BBox(*b), orientation=_infer_orientation(b))
            for b in wall_boxes
        ],
        openings = [
            Opening(
                type              = o['type'],
                bbox              = BBox.from_list(o['bbox']),
                wall_side         = o.get('wall_side', 'unknown'),
                estimated_width_m = o.get('estimated_width_m'),
                confidence        = o.get('confidence', 0.9),
            )
            for o in raw_openings
        ],
        n_rooms       = vlm_result.get('n_rooms'),
        floor_area_m2 = vlm_result.get('floor_area_m2'),
        source        = 'model',
    )
    validate_handoff(contract, 'labeling_output')
    return contract


# ══════════════════════════════════════════════════════════════
# 真实训练流程（替换 _mock_run_training）
# ══════════════════════════════════════════════════════════════

def run_real_training(
    cfg:      PseudoLabelConfig,
    tl:       TraceLogger,
    warnings: List[str],
) -> dict:
    """
    真实训练：获取 GPU 独占锁 → 卸载所有推理模型 → 训练。
    """
    from resource_manager import GPU_LOCK, prepare_for_training
    from training.trainer import train_one_version

    tl.info(f'开始训练  version={cfg.dataset_version}')
    t0 = time.time()

    SCHEDULER.unload_all()

    with prepare_for_training(None, GPU_LOCK):
        result = train_one_version(cfg.dataset_version, base_cfg=cfg)

    elapsed = round(time.time() - t0, 2)
    tl.metric('best_val_iou', result.get('best_val_iou', 0), stage='training')
    tl.info(f'训练完成  {elapsed}s  best_val_iou={result.get("best_val_iou", 0):.4f}')

    flush_vram()
    return result


# ══════════════════════════════════════════════════════════════
# 真实 3D 重建（替换 _mock_run_reconstruction）
# ══════════════════════════════════════════════════════════════

def run_real_reconstruction(
    contract: ReconstructContract,
    cfg:      PseudoLabelConfig,
    tl:       TraceLogger,
    warnings: List[str],
) -> str:
    """
    真实 3D 重建：wall_boxes + openings → .glb 文件。
    trimesh 在 CPU 运行，不占 GPU 显存。
    """
    from reconstruct_3d import build_3d_model, match_openings_to_walls
    import numpy as np

    tl.info(f'开始 3D 重建  walls={len(contract.wall_boxes)}')
    t0 = time.time()

    wall_boxes = [tuple(w['bbox']) for w in contract.wall_boxes]

    # 构建 openings 格式（match_openings_to_walls 需要）
    opening_boxes  = np.array([o['bbox'] for o in contract.openings], dtype=np.float32) \
                     if contract.openings else np.zeros((0, 4))
    opening_labels = np.array(
        [1 if o['type'] == 'door' else 2 for o in contract.openings], dtype=np.int64
    ) if contract.openings else np.zeros(0, dtype=np.int64)

    matched_openings = match_openings_to_walls(wall_boxes, opening_boxes, opening_labels)

    out_dir  = os.path.join(cfg.pseudo_out_dir, 'reconstruct')
    os.makedirs(out_dir, exist_ok=True)
    glb_path = os.path.join(out_dir, f'{int(time.time())}.glb')

    scene = build_3d_model(
        wall_boxes  = wall_boxes,
        openings    = matched_openings,
        output_path = glb_path,
    )

    if scene is None:
        warnings.append('[3D重建] build_3d_model 返回 None（trimesh 可能未安装）')
    else:
        tl.info(f'3D 重建完成  {time.time()-t0:.1f}s  → {glb_path}')

    return glb_path


# ══════════════════════════════════════════════════════════════
# orchestrator 集成入口
# ══════════════════════════════════════════════════════════════

def patch_orchestrator():
    """
    把 orchestrator.py 里的 mock 函数替换成真实实现。
    在 orchestrator.run_pipeline(use_mock=False) 调用前调用。

    用法：
        from skeleon.integration import patch_orchestrator
        patch_orchestrator()
        from orchestrator import run_pipeline
        run_pipeline(image_path, use_mock=False)
    """
    import orchestrator as orch

    def _real_labeling(image_path, cfg):
        warnings = []
        tl = TraceLogger('integration', '/tmp/trace_logs')
        tl.start() if hasattr(tl, 'start') else None
        return run_real_labeling(image_path, cfg, tl, warnings)

    def _real_training(contract, cfg):
        warnings = []
        tl = TraceLogger('integration', '/tmp/trace_logs')
        return run_real_training(cfg, tl, warnings)

    def _real_reconstruction(contract, cfg):
        warnings = []
        tl = TraceLogger('integration', '/tmp/trace_logs')
        return run_real_reconstruction(contract, cfg, tl, warnings)

    orch._mock_run_labeling      = _real_labeling
    orch._mock_run_training      = _real_training
    orch._mock_run_reconstruction = _real_reconstruction

    logger.info('[integration] orchestrator mock 函数已替换为真实实现')


# ══════════════════════════════════════════════════════════════
# 内部工具
# ══════════════════════════════════════════════════════════════

def _infer_orientation(bbox: tuple) -> str:
    """根据 bbox 宽高比推断墙体朝向。"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w == 0 and h == 0:
        return 'unknown'
    if w >= h * 2:
        return 'horizontal'
    if h >= w * 2:
        return 'vertical'
    return 'unknown'


def _make_mask_image_content(image_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """把 mask 叠加到原图，转成 VLM 的 base64 image content。"""
    import base64
    import io
    from PIL import Image as PILImage

    vis = image_rgb.copy()
    vis[mask == 1] = (vis[mask == 1] * 0.5 + np.array([0, 180, 0]) * 0.5).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(vis).save(buf, format='PNG')
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64}}


def _det_boxes_to_openings(det_boxes: np.ndarray, det_labels: np.ndarray) -> dict:
    """检测框转成 VLM 输出格式（VLM 失败时的 fallback）。"""
    openings = []
    for b, l in zip(det_boxes, det_labels):
        openings.append({
            'type':              'door' if int(l) == 1 else 'window',
            'bbox':              [int(v) for v in b],
            'wall_side':         'unknown',
            'estimated_width_m': None,
            'confidence':        0.8,
        })
    return {'openings': openings, 'n_rooms': None, 'floor_area_m2': None}
