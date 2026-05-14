# pipeline.py
import json
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.ops import nms as tv_nms

from config import CFG, DEVICE, logger, PseudoLabelConfig
from tools.sam2_refine import refine_mask_with_sam2
from tools.vlm_completion import vlm_semantic_completion
from tools.svg_export import generate_svg
from evaluation import evaluate, EvalWeights, EvalResult
from thought_logger import ThoughtLogger
from failure_rag import FailureRAG


# ══════════════════════════════════════════════════════════════
# Step 1：DINOv2+LoRA 滑动窗口推理
# ══════════════════════════════════════════════════════════════

def run_inference(
    image: np.ndarray,
    model,
    cfg: PseudoLabelConfig = CFG,
) -> dict:
    """
    DINOv2+LoRA 滑动窗口推理，生成初始 wall mask 和门窗 bbox。
    返回: {wall_mask, boxes, scores, labels}
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.norm_mean, cfg.norm_std),
    ])

    H, W   = image.shape[:2]
    ts     = cfg.tile_size
    stride = ts - cfg.tile_overlap

    ys = list(range(0, max(H - ts + 1, 1), stride))
    xs = list(range(0, max(W - ts + 1, 1), stride))
    if not ys or ys[-1] + ts < H:
        ys.append(max(H - ts, 0))
    if not xs or xs[-1] + ts < W:
        xs.append(max(W - ts, 0))

    wall_prob = np.zeros((H, W), dtype=np.float32)
    wall_cnt  = np.zeros((H, W), dtype=np.float32)
    all_boxes, all_scores, all_labels = [], [], []

    model.eval()
    for ty in ys:
        for tx in xs:
            tile   = image[ty:ty + ts, tx:tx + ts].copy()
            th, tw = tile.shape[:2]
            if th < ts or tw < ts:
                tile = cv2.copyMakeBorder(
                    tile, 0, ts - th, 0, ts - tw, cv2.BORDER_REFLECT_101
                )
            tile   = cv2.resize(tile, (ts, ts))
            tensor = tf(tile).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = model(tensor)

            prob = torch.softmax(out['seg_logits'], dim=1)[0, 1].cpu().numpy()
            prob = cv2.resize(prob, (tw, th))
            wall_prob[ty:ty + th, tx:tx + tw] += prob[:th, :tw]
            wall_cnt[ty:ty + th,  tx:tx + tw] += 1

            sx, sy = tw / ts, th / ts
            for det in out['det_outputs']:
                bxs  = det['boxes'].cpu().numpy()
                scrs = det['scores'].cpu().numpy()
                lbls = det['labels'].cpu().numpy()
                for b, s, l in zip(bxs, scrs, lbls):
                    if s >= 0.5:
                        all_boxes.append([
                            b[0] * sx + tx, b[1] * sy + ty,
                            b[2] * sx + tx, b[3] * sy + ty,
                        ])
                        all_scores.append(float(s))
                        all_labels.append(int(l))

    wall_prob   /= np.maximum(wall_cnt, 1)
    initial_mask = (wall_prob > 0.5).astype(np.uint8)

    if all_boxes:
        boxes_t  = torch.tensor(all_boxes,  dtype=torch.float32)
        scores_t = torch.tensor(all_scores, dtype=torch.float32)
        keep       = tv_nms(boxes_t, scores_t, 0.5)
        det_boxes  = boxes_t[keep].numpy()
        det_scores = scores_t[keep].numpy()
        det_labels = np.array(all_labels, dtype=np.int64)[keep.numpy()]
    else:
        det_boxes  = np.zeros((0, 4), dtype=np.float32)
        det_scores = np.zeros(0,      dtype=np.float32)
        det_labels = np.zeros(0,      dtype=np.int64)

    logger.info(
        f'run_inference 完成  '
        f'wall%={initial_mask.mean() * 100:.1f}%  '
        f'det_boxes={len(det_boxes)}'
    )
    return {'wall_mask': initial_mask, 'boxes': det_boxes,
            'scores': det_scores, 'labels': det_labels}


# ══════════════════════════════════════════════════════════════
# 完整四步流水线（接入 evaluation + ThoughtLogger + FailureRAG）
# ══════════════════════════════════════════════════════════════

def run_pseudo_label_pipeline(
    image_path:     str,
    cfg:            PseudoLabelConfig,
    dinov2_model,
    sam2_predictor,
    vlm_client,
    dry_run_vlm:    bool                  = True,
    gt_mask:        Optional[np.ndarray]  = None,
    eval_weights:   Optional[EvalWeights] = None,
    failure_rag:    Optional[FailureRAG]  = None,
    log_dir:        str                   = "./thought_logs",
) -> dict:
    """
    单张公司图纸的完整四步伪标注生成流水线。

    新增参数：
        gt_mask      : GT mask（有标注时传入，用于 IoU 计算；None=无监督模式）
        eval_weights : 三维评分权重（None 时用默认 0.5/0.3/0.2）
        failure_rag  : FailureRAG 实例（None 时不记录错题集）
        log_dir      : ThoughtLogger 写出报告的目录
    """
    from vector_logic import vectorize_wall_mask
    from postprocess_config import VectorizationConfig

    name    = Path(image_path).stem
    out_dir = os.path.join(cfg.pseudo_out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    t0      = time.time()

    logger.info(f'开始处理: {name}')

    # ── ThoughtLogger 启动 ──
    tl = ThoughtLogger(image_path, log_dir)
    tl.start()

    # ── 读图 ──
    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f'图片不存在: {image_path}'
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W    = img_rgb.shape[:2]

    # ════════════════════════════════════════
    # Step 1: DINOv2+LoRA 推理
    # ════════════════════════════════════════
    infer_out    = run_inference(img_rgb, dinov2_model, cfg)
    initial_mask = infer_out['wall_mask']
    det_boxes    = infer_out['boxes']
    det_labels   = infer_out['labels']

    tl.log_step(
        step_id=0, state="ACTING",
        reasoning="用 DINOv2+LoRA 生成初始 wall mask 和门窗 bbox",
        plan=["滑动窗口推理", "NMS 过滤检测框"],
        tool_choice="run_inference",
        confidence=0.9,
        success=True,
        metrics={
            "wall_coverage": round(float(initial_mask.mean()), 4),
            "det_boxes":     int(len(det_boxes)),
        },
    )

    # ════════════════════════════════════════
    # Step 2: SAM2 精化
    # ════════════════════════════════════════
    if sam2_predictor is not None:
        refined_mask = refine_mask_with_sam2(img_rgb, initial_mask, sam2_predictor, cfg)
        sam2_success = True
        logger.info(f'Step2 完成  refined_pixels={refined_mask.sum()}')
    else:
        refined_mask = initial_mask
        sam2_success = False
        logger.info('Step2 跳过（SAM2 未安装）')

    tl.log_step(
        step_id=1, state="ACTING",
        reasoning="SAM2 Point Prompt 精化 mask 边界",
        plan=["采样正负点", "SAM2 predict", "选最优候选"],
        tool_choice="refine_mask_with_sam2",
        confidence=0.85,
        success=sam2_success,
        metrics={"refined_pixels": int(refined_mask.sum())},
        failure_reason=None if sam2_success else "SAM2 未安装，跳过",
    )

    # ════════════════════════════════════════
    # Step 3: VLM 语义补全
    # ════════════════════════════════════════
    if dry_run_vlm or vlm_client is None:
        vlm_meta = {'openings': [], '_dry_run': True}
        for b, l in zip(det_boxes, det_labels):
            vlm_meta['openings'].append({
                'type':              'door' if l == 1 else 'window',
                'bbox':              [int(v) for v in b],
                'wall_side':         'unknown',
                'estimated_width_m': None,
                'confidence':        0.9,
            })
        vlm_success = True
        logger.info(f'Step3 dry_run  openings={len(vlm_meta["openings"])}')
    else:
        vlm_meta    = vlm_semantic_completion(img_rgb, refined_mask, cfg, vlm_client)
        vlm_success = True
        logger.info(f'Step3 完成  openings={len(vlm_meta["openings"])}')

    tl.log_step(
        step_id=2, state="ACTING",
        reasoning="VLM 识别门窗类型、朝向、尺寸",
        plan=["构造图像+mask prompt", "解析 JSON 输出"],
        tool_choice="vlm_semantic_completion",
        confidence=0.8,
        success=vlm_success,
        metrics={
            "n_openings": len(vlm_meta["openings"]),
            "dry_run":    int(dry_run_vlm),
        },
    )

    # ════════════════════════════════════════
    # Step 4: 矢量化 + SVG 生成
    # ════════════════════════════════════════
    vect_cfg   = VectorizationConfig(
        iou_threshold    = cfg.shrink_iou_thresh,
        min_segment_area = cfg.min_segment_area,
    )
    wall_boxes = vectorize_wall_mask(refined_mask, vect_cfg)
    logger.info(f'Step4 矢量化  wall_boxes={len(wall_boxes)}')

    svg_path = os.path.join(out_dir, 'pseudo_label.svg')
    generate_svg(
        image_wh    = (W, H),
        wall_boxes  = wall_boxes,
        openings    = vlm_meta['openings'],
        cfg         = cfg,
        output_path = svg_path,
    )

    tl.log_step(
        step_id=3, state="ACTING",
        reasoning="矢量化 mask 为 wall_boxes，生成 CubiCasa 兼容 SVG",
        plan=["Shrinking 算法", "generate_svg"],
        tool_choice="generate_svg",
        confidence=0.95,
        success=True,
        metrics={"n_wall_boxes": len(wall_boxes), "svg_written": 1},
    )

    # ── 保存中间结果 ──
    cv2.imwrite(os.path.join(out_dir, 'initial_mask.png'), initial_mask * 255)
    cv2.imwrite(os.path.join(out_dir, 'refined_mask.png'), refined_mask * 255)
    with open(os.path.join(out_dir, 'semantic_meta.json'), 'w') as f:
        json.dump(vlm_meta, f, indent=2, ensure_ascii=False)

    # ════════════════════════════════════════
    # 综合评分
    # ════════════════════════════════════════
    _gt_mask = gt_mask if gt_mask is not None else np.zeros((H, W), dtype=np.uint8)

    eval_result: EvalResult = evaluate(
        pred_mask   = refined_mask,
        gt_mask     = _gt_mask,
        wall_boxes  = wall_boxes,
        openings    = vlm_meta['openings'],
        image_rgb   = img_rgb,
        cfg         = cfg,
        weights     = eval_weights,
        vlm_client  = vlm_client if not dry_run_vlm else None,
    )

    # ── ThoughtLogger：写报告 ──
    tl.log_eval(eval_result.to_dict())
    log_dict      = tl.finish()
    log_json_path = os.path.join(log_dir, f"{name}_thought_log.json")

    # 把评分报告也存到 out_dir，方便批量汇总
    with open(os.path.join(out_dir, 'eval_result.json'), 'w') as f:
        json.dump(eval_result.to_dict(), f, indent=2, ensure_ascii=False)

    # ── FailureRAG：失败时记录错题 ──
    if failure_rag is not None and not eval_result.passed:
        failure_rag.add(
            image_name       = name,
            situation        = (
                f"pipeline 失败  "
                f"IoU={eval_result.iou_pixel:.3f}  "
                f"Topo={eval_result.c_topological:.3f}  "
                f"walls={len(wall_boxes)}  "
                f"openings={len(vlm_meta['openings'])}"
            ),
            eval_result      = eval_result.to_dict(),
            thought_log_path = log_json_path,
        )

    elapsed = round(time.time() - t0, 2)
    doors   = [o for o in vlm_meta['openings'] if o['type'] == 'door']
    windows = [o for o in vlm_meta['openings'] if o['type'] == 'window']

    logger.info(
        f'完成  {elapsed}s  '
        f'walls={len(wall_boxes)}  '
        f'doors={len(doors)}  '
        f'windows={len(windows)}  '
        f'S_total={eval_result.s_total:.3f}  '
        f'{"✓" if eval_result.passed else "✗"}'
    )

    return {
        'image_path':   image_path,
        'svg_path':     svg_path,
        'wall_boxes':   wall_boxes,
        'openings':     vlm_meta['openings'],
        'eval':         eval_result.to_dict(),
        'thought_log':  log_json_path,
        'metrics': {
            'n_walls':   len(wall_boxes),
            'n_doors':   len(doors),
            'n_windows': len(windows),
            'elapsed':   elapsed,
            's_total':   eval_result.s_total,
            'passed':    eval_result.passed,
        },
    }
