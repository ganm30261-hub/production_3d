# orchestrator.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

指挥中心：支柱总集成 —— 串联所有 Agent

把五个支柱整合成一个可运行的系统：
    全局状态    → global_state.PipelineState
    数据契约    → contracts.LabelContract / TrainingContract / ReconstructContract
    可观测性    → observability.TraceLogger / traced_stage
    资源调度    → resource_manager.GPU_LOCK / MODEL_REGISTRY
    工具接口    → agent_tools.TOOL_REGISTRY

骨架期"完成"的标准（可以直接跑通）：
    日志显示：Labeler 启动 → 伪造标注 → Trainer 启动 → 模拟训练 → Architect 启动 → 空 3D 模型
    GCS 上出现预期的文件夹结构
    整个过程没有报错
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import dataclasses
import os
import time
from typing import List, Optional

from config import CFG, logger, PseudoLabelConfig
from skeleon.global_state import PipelineState, Stage, StageResult, TransitionResult
from skeleon.contracts import (
    LabelContract, TrainingContract, ReconstructContract,
    validate_handoff, ValidationError,
)
from skeleon.observability import TraceLogger, traced_stage, classify_error, ErrorCategory
from skeleon.resource_manager import GPU_LOCK, MODEL_REGISTRY, prepare_for_training


# ══════════════════════════════════════════════════════════════
# 骨架期 Mock 工具（等模型训好直接替换）
# ══════════════════════════════════════════════════════════════

def _mock_run_labeling(image_path: str, cfg: PseudoLabelConfig) -> LabelContract:
    """
    Mock 标注：暂时返回假数据，格式与真实输出完全一致。
    等 DINOv2+LoRA 训练收敛后，替换为真实 pipeline.run_pseudo_label_pipeline()。
    """
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ValidationError(f'无法读取图片: {image_path}')
    H, W = img.shape[:2]
    label = LabelContract.mock(image_path, image_wh=(W, H))
    label.eval_scores = {'iou_pixel': 0.0, 's_total': 0.0, 'note': 'mock'}
    return label


def _mock_run_training(contract: TrainingContract, cfg: PseudoLabelConfig) -> dict:
    """
    Mock 训练：模拟训练过程，返回假的训练结果。
    等真实训练完成后，替换为 training.trainer.train_one_version()。
    """
    logger.info(f'[mock] 模拟训练 n_samples={contract.n_samples}  睡眠 2s...')
    time.sleep(2)
    return {
        'best_val_iou': 0.0,
        'best_ckpt':    None,
        'note':         'mock',
    }


def _mock_run_reconstruction(contract: ReconstructContract, cfg: PseudoLabelConfig) -> str:
    """
    Mock 3D 重建：生成一个空的 .glb 占位文件。
    等 reconstruct_3d.build_3d_model() 就绪后替换。
    """
    import tempfile
    out_dir  = os.path.join(cfg.pseudo_out_dir, 'reconstruct')
    os.makedirs(out_dir, exist_ok=True)
    glb_path = os.path.join(out_dir, f'mock_{int(time.time())}.glb')
    with open(glb_path, 'wb') as f:
        f.write(b'glTF')   # GLB magic bytes，合法的最小 GLB
    logger.info(f'[mock] 生成空 GLB: {glb_path}')
    return glb_path


# ══════════════════════════════════════════════════════════════
# 三个 Agent 的执行逻辑
# ══════════════════════════════════════════════════════════════

def run_labeling_agent(
    state:    PipelineState,
    tl:       TraceLogger,
    cfg:      PseudoLabelConfig,
    use_mock: bool = True,
) -> LabelContract:
    """
    标注 Agent：图片 → LabelContract

    use_mock=True  骨架期用假数据
    use_mock=False 接入真实 pipeline（模型训好后用）
    """
    t0 = time.time()
    with traced_stage(tl, 'labeling') as ctx:

        if use_mock:
            label = _mock_run_labeling(state.image_path, cfg)
        else:
            # 真实路径：等模型训好后取消注释
            from pipeline import run_pseudo_label_pipeline
            from tools.sam2_refine import load_sam2_predictor
            import anthropic

            with MODEL_REGISTRY.use('dinov2', loader=lambda: _load_dinov2(cfg)):
                result = run_pseudo_label_pipeline(
                    image_path     = state.image_path,
                    cfg            = cfg,
                    dinov2_model   = MODEL_REGISTRY._models['dinov2'],
                    sam2_predictor = None,
                    vlm_client     = anthropic.Anthropic(),
                    dry_run_vlm    = False,
                )
            label = _result_to_label_contract(result)

        # 校验契约
        validate_handoff(label, 'labeling_output')

        # 保存到 staging
        meta_dir  = os.path.join(cfg.pseudo_out_dir, 'labels')
        os.makedirs(meta_dir, exist_ok=True)
        meta_path = os.path.join(meta_dir, f'{state.trace_id}_label.json')
        label.save_meta(meta_path)

        elapsed = round(time.time() - t0, 2)
        ctx.set_result('success')

    tl.metric('n_wall_boxes', len(label.wall_boxes),  stage='labeling')
    tl.metric('n_openings',   len(label.openings),    stage='labeling')
    tl.metric('elapsed_s',    elapsed,                stage='labeling')

    state.transition(
        StageResult.success(
            metrics      = {'n_walls': len(label.wall_boxes), 'n_openings': len(label.openings)},
            output_paths = {'label_meta': meta_path},
            elapsed_s    = elapsed,
        ),
        Stage.LABELED,
    )
    return label


def run_training_agent(
    state:    PipelineState,
    tl:       TraceLogger,
    cfg:      PseudoLabelConfig,
    label:    LabelContract,
    use_mock: bool = True,
) -> dict:
    """
    训练 Agent：LabelContract → 更新后的模型权重

    训练是独占模式，获取 GPU 锁后才能启动。
    """
    t0 = time.time()
    state.transition(StageResult.success(), Stage.TRAINING)

    # 构建训练契约
    train_root = os.path.join(cfg.pseudo_out_dir, 'train_data')
    train_contract = TrainingContract(
        train_root      = train_root,
        dataset_version = cfg.dataset_version,
        n_samples       = 1,   # 单图模式，批量时修改
        label_source    = label.source,
    )

    with traced_stage(tl, 'training') as ctx:
        if use_mock:
            train_result = _mock_run_training(train_contract, cfg)
        else:
            with prepare_for_training(MODEL_REGISTRY, GPU_LOCK):
                from training.trainer import train_one_version
                train_result = train_one_version(cfg.dataset_version, base_cfg=cfg)

        elapsed = round(time.time() - t0, 2)
        ctx.set_result('success')

    tl.metric('best_val_iou', train_result.get('best_val_iou', 0), stage='training')
    tl.metric('elapsed_s',    elapsed, stage='training')

    state.transition(
        StageResult.success(
            metrics      = train_result,
            output_paths = {'best_ckpt': train_result.get('best_ckpt', '')},
            elapsed_s    = elapsed,
        ),
        Stage.TRAINED,
    )
    return train_result


def run_reconstruction_agent(
    state:    PipelineState,
    tl:       TraceLogger,
    cfg:      PseudoLabelConfig,
    label:    LabelContract,
    use_mock: bool = True,
) -> str:
    """
    3D 重建 Agent：LabelContract → .glb 文件
    """
    t0 = time.time()
    state.transition(StageResult.success(), Stage.RECONSTRUCTING)

    recon = ReconstructContract.from_label(label, pixels_per_meter=cfg.pixels_per_meter)
    validate_handoff(recon, 'reconstruction_input')

    with traced_stage(tl, 'reconstruction') as ctx:
        if use_mock:
            glb_path = _mock_run_reconstruction(recon, cfg)
        else:
            with MODEL_REGISTRY.use('trimesh', loader=lambda: None, required_mb=500):
                from reconstruct_3d import build_3d_model, match_openings_to_walls
                import numpy as np
                wall_boxes = [tuple(w['bbox']) for w in recon.wall_boxes]
                openings   = recon.openings
                out_path   = os.path.join(cfg.pseudo_out_dir, f'{state.trace_id}.glb')
                build_3d_model(wall_boxes, openings, output_path=out_path)
                glb_path   = out_path

        elapsed = round(time.time() - t0, 2)
        ctx.set_result('success')

    tl.metric('elapsed_s', elapsed, stage='reconstruction')

    state.transition(
        StageResult.success(
            output_paths = {'glb_path': glb_path},
            elapsed_s    = elapsed,
        ),
        Stage.DONE,
    )
    return glb_path


# ══════════════════════════════════════════════════════════════
# 主编排入口
# ══════════════════════════════════════════════════════════════

def run_pipeline(
    image_path:   str,
    cfg:          PseudoLabelConfig = None,
    use_mock:     bool              = True,
    resume_trace: Optional[str]     = None,
    log_dir:      str               = '/tmp/trace_logs',
) -> dict:
    """
    完整流水线主入口：图片 → 标注 → 训练 → 3D 重建

    参数：
        image_path   : 原始图片路径（本地或 GCS）
        cfg          : 配置（None 时用默认 CFG）
        use_mock     : True = 骨架期 mock，False = 接入真实模型
        resume_trace : 断点续跑时传入之前的 trace_id
        log_dir      : Trace 日志输出目录

    返回：
        {
            trace_id, stage, outputs,
            warnings, elapsed_s, success
        }
    """
    cfg = cfg or CFG
    t0  = time.time()

    # ── 创建或恢复状态 ──
    if resume_trace:
        state = PipelineState.load(resume_trace)
        logger.info(f'断点续跑: trace_id={resume_trace}  stage={state.stage.value}')
    else:
        state = PipelineState.create(image_path)

    tl = TraceLogger(state.trace_id, log_dir)
    tl.info(f'流水线启动  image={image_path}  mock={use_mock}')

    label        = None
    train_result = None
    glb_path     = None

    try:
        # ── Stage 1: 标注（幂等，已完成则跳过）──
        if not state.is_stage_done(Stage.LABELING):
            tl.info('启动标注 Agent')
            label = run_labeling_agent(state, tl, cfg, use_mock)
        else:
            tl.info('标注已完成，跳过')
            meta_path = state.outputs.get('label_meta')
            if meta_path and os.path.exists(meta_path):
                label = LabelContract.load_meta(meta_path)

        if label is None:
            raise ValidationError('标注结果为空，无法继续')

        # ── Stage 2: 训练（幂等）──
        if not state.is_stage_done(Stage.TRAINING):
            tl.info('启动训练 Agent')
            train_result = run_training_agent(state, tl, cfg, label, use_mock)
        else:
            tl.info('训练已完成，跳过')

        # ── Stage 3: 3D 重建（幂等）──
        if not state.is_stage_done(Stage.RECONSTRUCTING):
            tl.info('启动 3D 重建 Agent')
            glb_path = run_reconstruction_agent(state, tl, cfg, label, use_mock)
        else:
            tl.info('3D 重建已完成，跳过')
            glb_path = state.outputs.get('glb_path')

    except ValidationError as e:
        classified = classify_error(e, state.trace_id)
        tl.error(f'数据校验失败: {e}')
        state.add_warning(str(e))
        if state.can_retry():
            state.transition(StageResult.retry(str(e)), Stage.RETRYING)
        else:
            state.transition(StageResult.fail(str(e)), Stage.FAILED)

    except Exception as e:
        classified = classify_error(e, state.trace_id)
        tl.error(f'流水线异常: {classified.category.value}  {e}', exc=e)

        if classified.category == ErrorCategory.HARDWARE and state.can_retry():
            # 显存 OOM：清理后重试
            from resource_manager import clear_gpu_memory
            clear_gpu_memory()
            state.transition(StageResult.retry(f'HardwareError: {e}'), Stage.RETRYING)
        elif classified.category == ErrorCategory.SCRIPT:
            # 代码 bug：直接 FAILED，不重试
            state.transition(StageResult.fail(f'ScriptError: {e}'), Stage.FAILED)
            raise   # ScriptError 重新抛出，让开发者看到
        else:
            state.transition(StageResult.fail(str(e)), Stage.FAILED)

    finally:
        trace_path = tl.finish()
        state.set_output('trace_log', trace_path)

    elapsed = round(time.time() - t0, 2)
    success = state.stage == Stage.DONE

    logger.info(
        f'[{state.trace_id}] 流水线{"完成 ✓" if success else "失败 ✗"}  '
        f'elapsed={elapsed}s  stage={state.stage.value}'
    )

    return {
        'trace_id': state.trace_id,
        'stage':    state.stage.value,
        'success':  success,
        'outputs':  state.outputs,
        'warnings': state.warnings,
        'elapsed_s': elapsed,
    }


# ══════════════════════════════════════════════════════════════
# 批量处理入口
# ══════════════════════════════════════════════════════════════

def run_batch(
    image_paths: List[str],
    cfg:         PseudoLabelConfig = None,
    use_mock:    bool = True,
    log_dir:     str  = '/tmp/trace_logs',
) -> List[dict]:
    """
    批量处理多张图片，每张独立 trace_id，单张失败不影响其他。
    """
    cfg     = cfg or CFG
    results = []

    for i, path in enumerate(image_paths):
        logger.info(f'批量处理 [{i+1}/{len(image_paths)}]: {path}')
        try:
            result = run_pipeline(path, cfg=cfg, use_mock=use_mock, log_dir=log_dir)
        except Exception as e:
            # ScriptError 才会走到这里，记录后继续
            result = {'image_path': path, 'success': False, 'error': str(e)}
        results.append(result)

    n_ok   = sum(1 for r in results if r.get('success'))
    n_fail = len(results) - n_ok
    logger.info(f'批量完成  成功={n_ok}  失败={n_fail}  总计={len(results)}')
    return results


# ══════════════════════════════════════════════════════════════
# 内部工具
# ══════════════════════════════════════════════════════════════

def _load_dinov2(cfg: PseudoLabelConfig):
    """加载真实 DINOv2+LoRA 模型（非 mock 路径用）。"""
    import torch, glob
    from model.model import DINOv2LoRAModel
    from config import DEVICE

    model = DINOv2LoRAModel(cfg).to(DEVICE)
    ckpt_dir = cfg.checkpoint_dir
    matches  = sorted(glob.glob(os.path.join(ckpt_dir, '*_best.pth')))
    if not matches:
        raise FileNotFoundError(f'找不到 checkpoint: {ckpt_dir}')
    ckpt = torch.load(matches[-1], map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


def _result_to_label_contract(result: dict) -> LabelContract:
    """把 pipeline.run_pseudo_label_pipeline() 的输出转成 LabelContract。"""
    from contracts import WallBox, Opening, BBox
    wall_boxes = [
        WallBox(bbox=BBox(*[int(v) for v in b]))
        for b in result.get('wall_boxes', [])
    ]
    openings = [
        Opening(
            type      = o['type'],
            bbox      = BBox.from_list(o['bbox']),
            wall_side = o.get('wall_side', 'unknown'),
            confidence = o.get('confidence', 0.9),
        )
        for o in result.get('openings', [])
    ]
    import cv2
    img = cv2.imread(result['image_path'])
    H, W = img.shape[:2] if img is not None else (0, 0)

    return LabelContract(
        image_path  = result['image_path'],
        image_wh    = (W, H),
        wall_boxes  = wall_boxes,
        openings    = openings,
        eval_scores = result.get('eval', {}),
        source      = 'model',
    )
