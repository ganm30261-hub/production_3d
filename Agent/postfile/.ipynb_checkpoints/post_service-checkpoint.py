"""
post_service.py — 逻辑层：后处理流水线调度

统领全局，定义对外的统一接口。按顺序调用：
  1. 模型推理（滑动窗口 + NMS）
  2. 矢量化 vector_logic.vectorize_wall_mask
  3. 门窗匹配 reconstruct_3d.match_openings_to_walls
  4. 3D 重建 reconstruct_3d.build_3d_model

验证系统（批量评估 + 失败案例分析）也在这里。

对外接口：
    run_inference(image, cfg)                     → 推理结果 dict
    run_postprocess(image_path, output_dir, cfg)  → 完整流水线结果
    validate_single_sample(folder, ...)           → 单样本逐阶段验证
    batch_validate(data_folder, n_samples, ...)   → 批量评估 + 汇总表

用法：
    python post_service.py --image floor.png --output_dir ./outputs
    python post_service.py --validate --n_samples 20
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from postprocess_config import (
    VectorizationConfig, InferenceConfig, ValidationTargets, StagingConfig,
    VECT_CFG, INF_CFG, VAL_TARGETS, STAGING_CFG,
    OUTPUT_DIR, STAGING_DIR, ARCHIVE_DIR,
)
from vector_logic import (
    vectorize_wall_mask, compute_vectorization_iou, wall_boxes_to_mask,
)
from reconstruct_3d import match_openings_to_walls, build_3d_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── 延迟 import 模型（不运行 validate-only 时可跳过）──
_model       = None
_img_transform = None
_device      = None


def _get_model():
    """懒加载模型（避免批量验证时也必须有 GPU）"""
    global _model, _img_transform, _device
    if _model is not None:
        return _model, _img_transform, _device

    import sys as _sys
    from config import CUBICASA_ROOT, CHECKPOINT_DIR

    _sys.path.insert(0, CUBICASA_ROOT)
    os.chdir(CUBICASA_ROOT)

    from model_arch import FloorplanModel

    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _model  = FloorplanModel.__new__(FloorplanModel)

    # 使用 postprocess_config 里的 InferenceConfig 参数
    from config import PaperConfig
    cfg = PaperConfig()

    from model_arch import build_model
    _model = build_model(cfg, _device)
    ckpt   = torch.load(f'{CHECKPOINT_DIR}/best_model.pth', map_location=_device)
    _model.load_state_dict(ckpt['model_state'])
    _model.eval()
    logger.info(f'模型加载完成  epoch={ckpt["epoch"]}  val_wall_iou={ckpt["val_wall_iou"]:.4f}')

    _img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return _model, _img_transform, _device


# ══════════════════════════════════════════════════════════════
# Section 2.1：滑动窗口推理
# ══════════════════════════════════════════════════════════════

def run_inference(
    image: np.ndarray,
    cfg:   InferenceConfig = INF_CFG,
) -> dict:
    """
    对单张 RGB 图进行完整推理（滑动窗口 + 结果融合 + NMS）

    返回：
        wall_mask  : (H, W) uint8
        wall_prob  : (H, W) float32
        boxes      : (N, 4) float32
        scores     : (N,) float32
        labels     : (N,) int64  1=door 2=window
    """
    model, img_transform, device = _get_model()

    h, w      = image.shape[:2]
    tile_size = cfg.tile_size
    overlap   = cfg.tile_overlap
    stride    = tile_size - overlap

    wall_prob = np.zeros((h, w), dtype=np.float32)
    wall_cnt  = np.zeros((h, w), dtype=np.float32)
    all_boxes, all_scores, all_labels = [], [], []

    ys = list(range(0, max(h - tile_size + 1, 1), stride))
    xs = list(range(0, max(w - tile_size + 1, 1), stride))
    if not ys or ys[-1] + tile_size < h: ys.append(max(h - tile_size, 0))
    if not xs or xs[-1] + tile_size < w: xs.append(max(w - tile_size, 0))

    for ty in ys:
        for tx in xs:
            tile    = image[ty:ty + tile_size, tx:tx + tile_size].copy()
            th, tw  = tile.shape[:2]

            if th < tile_size or tw < tile_size:
                tile = cv2.copyMakeBorder(
                    tile, 0, tile_size - th, 0, tile_size - tw,
                    cv2.BORDER_REFLECT_101,
                )
            tile   = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
            tensor = img_transform(tile).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(tensor)

            prob        = torch.softmax(out['seg_logits'], dim=1)[0, 1].cpu().numpy()
            prob_resize = cv2.resize(prob, (tw, th), interpolation=cv2.INTER_LINEAR)
            wall_prob[ty:ty + th, tx:tx + tw] += prob_resize[:th, :tw]
            wall_cnt[ty:ty + th,  tx:tx + tw] += 1

            det = out['det_outputs'][0]
            if len(det['boxes']) > 0:
                boxes   = det['boxes'].cpu().numpy()
                scores  = det['scores'].cpu().numpy()
                labels  = det['labels'].cpu().numpy()
                scale_x = tw / tile_size
                scale_y = th / tile_size
                keep    = scores >= cfg.det_score_thresh
                for box, score, label in zip(boxes[keep], scores[keep], labels[keep]):
                    all_boxes.append([
                        box[0] * scale_x + tx,
                        box[1] * scale_y + ty,
                        box[2] * scale_x + tx,
                        box[3] * scale_y + ty,
                    ])
                    all_scores.append(score)
                    all_labels.append(label)

    wall_prob /= np.maximum(wall_cnt, 1)
    wall_mask  = (wall_prob > 0.5).astype(np.uint8)

    if all_boxes:
        from torchvision.ops import nms
        boxes_t  = torch.tensor(all_boxes,  dtype=torch.float32)
        scores_t = torch.tensor(all_scores, dtype=torch.float32)
        labels_t = torch.tensor(all_labels, dtype=torch.int64)
        keep     = nms(boxes_t, scores_t, iou_threshold=cfg.nms_iou_thresh)
        all_boxes  = boxes_t[keep].numpy()
        all_scores = scores_t[keep].numpy()
        all_labels = labels_t[keep].numpy()
    else:
        all_boxes  = np.zeros((0, 4), dtype=np.float32)
        all_scores = np.zeros(0, dtype=np.float32)
        all_labels = np.zeros(0, dtype=np.int64)

    return {
        'wall_mask': wall_mask,
        'wall_prob': wall_prob,
        'boxes':     all_boxes,
        'scores':    all_scores,
        'labels':    all_labels,
    }


# ══════════════════════════════════════════════════════════════
# 完整端到端后处理 Pipeline
# ══════════════════════════════════════════════════════════════

def run_postprocess(
    image_path:    str,
    output_dir:    str          = OUTPUT_DIR,
    vect_cfg:      VectorizationConfig = VECT_CFG,
    inf_cfg:       InferenceConfig     = INF_CFG,
    export_3d:     bool         = True,
    task_id:       Optional[str] = None,
) -> dict:
    """
    完整端到端后处理流水线
    输入：平面图路径
    输出：result dict，并在 output_dir 写入结果文件

    result 格式（可直接序列化为 JSON）：
    {
        task_id, image_path, wall_boxes, doors, windows,
        glb_path, stats: {n_walls, n_doors, n_windows, elapsed}
    }
    """
    os.makedirs(output_dir, exist_ok=True)
    task_id = task_id or str(uuid.uuid4())[:8]
    name    = Path(image_path).stem
    t0      = time.time()

    logger.info(f'[{task_id}] 开始后处理: {name}')

    # ── Step 1：读取图片 ──
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f'图片不存在: {image_path}')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Step 2：推理（Section 2.1）──
    pred      = run_inference(img_rgb, inf_cfg)
    wall_mask = pred['wall_mask']
    logger.info(f'[{task_id}] 推理完成  wall%={wall_mask.mean()*100:.1f}%  det={len(pred["boxes"])}')

    # ── Step 3：矢量化（Section 2.4）──
    wall_boxes = vectorize_wall_mask(wall_mask, vect_cfg)
    logger.info(f'[{task_id}] 矢量化完成  wall_boxes={len(wall_boxes)}')

    # ── Step 4：门窗匹配（Section 2.5）──
    openings = match_openings_to_walls(wall_boxes, pred['boxes'], pred['labels'])
    doors    = [o for o in openings if o['type'] == 'door']
    windows  = [o for o in openings if o['type'] == 'window']

    # ── Step 5：3D 重建（Section 2.5）──
    glb_path = None
    if export_3d and wall_boxes:
        glb_path = os.path.join(output_dir, f'{name}_{task_id}_3d.glb')
        build_3d_model(wall_boxes, openings, output_path=glb_path)

    # ── Step 6：保存可视化结果图（无 plt，直接用 cv2）──
    _save_vis(image_path, wall_mask, wall_boxes, pred['boxes'], pred['labels'],
              os.path.join(output_dir, f'{name}_{task_id}_vis.png'))

    elapsed = round(time.time() - t0, 2)
    result  = {
        'task_id':    task_id,
        'image_path': image_path,
        'wall_boxes': [list(b) for b in wall_boxes],
        'doors':      [{'box': list(o['opening_box'])} for o in doors],
        'windows':    [{'box': list(o['opening_box'])} for o in windows],
        'glb_path':   glb_path,
        'stats': {
            'n_walls':   len(wall_boxes),
            'n_doors':   len(doors),
            'n_windows': len(windows),
            'elapsed':   elapsed,
        },
    }

    json_path = os.path.join(output_dir, f'{name}_{task_id}_result.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f'[{task_id}] 完成  {elapsed}s  walls={len(wall_boxes)}  '
                f'doors={len(doors)}  windows={len(windows)}')
    return result


def _save_vis(
    image_path: str,
    wall_mask:  np.ndarray,
    wall_boxes: list,
    det_boxes:  np.ndarray,
    det_labels: np.ndarray,
    out_path:   str,
):
    """保存矢量化 overlay 图（不依赖 matplotlib）"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return
    h, w = wall_mask.shape
    img  = cv2.resize(img_bgr, (w, h))

    overlay   = img.copy()
    wall_layer = np.zeros_like(overlay)
    wall_layer[wall_mask == 1] = [0, 180, 0]
    cv2.addWeighted(wall_layer, 0.35, overlay, 0.65, 0, overlay)

    for b in wall_boxes:
        cv2.rectangle(overlay, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    for b, lbl in zip(det_boxes, det_labels):
        color = (255, 50, 50) if int(lbl) == 1 else (50, 50, 255)
        cv2.rectangle(overlay, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)

    cv2.imwrite(out_path, overlay)
    logger.info(f'可视化已保存: {out_path}')


# ══════════════════════════════════════════════════════════════
# 验证系统
# ══════════════════════════════════════════════════════════════

def validate_single_sample(
    folder:     str,
    data_folder: str,
    cubicasa_root: str,
    vect_cfg:   VectorizationConfig = VECT_CFG,
    inf_cfg:    InferenceConfig     = INF_CFG,
    targets:    ValidationTargets   = VAL_TARGETS,
) -> dict:
    """
    对单个 CubiCasa5k 样本做完整后处理并逐阶段验证

    返回 metrics dict（键名与 batch_validate 的 summary 一致）
    内部结果（_前缀字段）用于可视化，批量评估时可丢弃。
    """
    import sys as _sys
    _sys.path.insert(0, cubicasa_root)
    from floortrans.loaders.house import House
    from torchvision.ops import box_iou as tv_box_iou

    folder   = folder.strip('/')
    img_path = os.path.join(data_folder, folder, 'F1_scaled.png')
    svg_path = os.path.join(data_folder, folder, 'model.svg')

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return {'folder': folder, 'status': 'no_image'}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]

    # 加载 GT
    try:
        house   = House(svg_path, h, w)
        seg     = house.get_segmentation_tensor()
        gt_wall = (seg[0] == 2).astype(np.uint8)
    except Exception as e:
        return {'folder': folder, 'status': f'svg_error: {e}'}

    # GT 门窗 bbox
    gt_boxes_list, gt_labels_list = [], []
    for cls_id, lbl in [(2, 1), (1, 2)]:
        m = (seg[1] == cls_id).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < 100:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            gt_boxes_list.append([x, y, x + bw, y + bh])
            gt_labels_list.append(lbl)
    gt_boxes  = np.array(gt_boxes_list,  dtype=np.float32) if gt_boxes_list else np.zeros((0, 4))
    gt_labels = np.array(gt_labels_list, dtype=np.int64)   if gt_labels_list else np.zeros(0, dtype=np.int64)

    # 裁剪到 wall bbox（与训练保持一致）
    img_crop, gt_crop = _crop_to_wall(img_rgb, gt_wall)
    if img_crop is None:
        return {'folder': folder, 'status': 'no_wall'}

    # 推理
    pred      = run_inference(img_crop, inf_cfg)
    wall_pred = pred['wall_mask']
    if wall_pred.shape != gt_crop.shape:
        wall_pred = cv2.resize(
            wall_pred.astype(np.float32),
            (gt_crop.shape[1], gt_crop.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)

    # 验证 1：Wall Mask IoU
    iou_mask = _iou_mask(wall_pred, gt_crop)

    # 验证 2：矢量化 IoU
    wall_boxes = vectorize_wall_mask(wall_pred, vect_cfg)
    vect_m     = wall_boxes_to_mask(wall_boxes, gt_crop.shape)
    iou_vect   = _iou_mask(vect_m, gt_crop)
    vect_pixel_ratio = float(vect_m.sum()) / (float(wall_pred.sum()) + 1e-8)

    # 验证 3：门窗检测 P/R/F1
    tp, fp, fn = _det_metrics(
        pred['boxes'], pred['scores'], pred['labels'],
        gt_boxes, gt_labels,
        iou_thresh=targets.det_iou_thresh,
        score_thresh=targets.det_score_thresh,
    )
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)

    # 验证 4：门窗落墙匹配率
    door_m, door_t, win_m, win_t = _match_rate(
        pred['boxes'], pred['labels'], wall_pred,
        overlap_thresh=targets.match_overlap_thresh,
    )
    door_match = door_m / (door_t + 1e-8)
    win_match  = win_m  / (win_t  + 1e-8)

    # 验证 5：3D 几何合理性（bbox 宽高比）
    openings = match_openings_to_walls(wall_boxes, pred['boxes'], pred['labels'])
    aspect_ok = sum(
        1 for b in wall_boxes
        if 0.1 < (b[2] - b[0]) / (b[3] - b[1] + 1e-8) < 10
    )
    geom_ok_rate = aspect_ok / (len(wall_boxes) + 1e-8)

    # 日志输出（DEBUG 级别，批量时不刷屏）
    logger.debug(
        f'{folder}  mask={iou_mask:.3f}  vect={iou_vect:.3f}  '
        f'f1={f1:.3f}  door_match={door_match:.3f}'
    )

    return {
        'folder':             folder,
        'status':             'ok',
        'iou_mask':           iou_mask,
        'iou_vect':           iou_vect,
        'n_wall_boxes':       len(wall_boxes),
        'vect_pixel_ratio':   vect_pixel_ratio,
        'det_precision':      prec,
        'det_recall':         rec,
        'det_f1':             f1,
        'door_match':         door_match,
        'win_match':          win_match,
        'n_doors_3d':         len([o for o in openings if o['type'] == 'door']),
        'n_windows_3d':       len([o for o in openings if o['type'] == 'window']),
        'geom_ok_rate':       geom_ok_rate,
        # 中间结果（用于可视化，批量时 strip）
        '_img_crop':   img_crop,
        '_gt_crop':    gt_crop,
        '_wall_pred':  wall_pred,
        '_vect_m':     vect_m,
        '_wall_boxes': wall_boxes,
        '_det_boxes':  pred['boxes'],
        '_det_labels': pred['labels'],
    }


def batch_validate(
    data_folder:    str,
    cubicasa_root:  str,
    n_samples:      int  = 400,
    save_dir:       Optional[str] = None,
    vect_cfg:       VectorizationConfig = VECT_CFG,
    inf_cfg:        InferenceConfig     = INF_CFG,
    targets:        ValidationTargets   = VAL_TARGETS,
) -> dict:
    """
    批量评估后处理质量（对应论文 Table 1 的评估方法）
    返回各指标的均值/标准差/分位数 summary dict
    """
    from numpy import genfromtxt

    val_folders = genfromtxt(os.path.join(data_folder, 'val.txt'), dtype='str')
    folders     = val_folders[:n_samples]

    records, skip = [], 0
    logger.info(f'开始批量验证  n={len(folders)}')

    for folder in folders:
        r = validate_single_sample(
            folder, data_folder, cubicasa_root, vect_cfg, inf_cfg, targets
        )
        if r.get('status') != 'ok':
            skip += 1
            continue
        records.append({k: v for k, v in r.items() if not k.startswith('_') and isinstance(v, (int, float))})

    n = len(records)
    logger.info(f'有效样本: {n}/{len(folders)}  跳过: {skip}')

    metrics = [
        'iou_mask', 'iou_vect', 'n_wall_boxes',
        'det_precision', 'det_recall', 'det_f1',
        'door_match', 'win_match', 'geom_ok_rate',
    ]
    summary = {}
    for m in metrics:
        vals = [r[m] for r in records if m in r]
        if not vals:
            continue
        a = np.array(vals)
        summary[m] = {
            'mean': float(a.mean()),
            'std':  float(a.std()),
            'p25':  float(np.percentile(a, 25)),
            'p50':  float(np.percentile(a, 50)),
            'p75':  float(np.percentile(a, 75)),
            'p10':  float(np.percentile(a, 10)),
        }

    # 打印汇总表
    paper_targets = {
        'iou_mask': 0.81, 'iou_vect': 0.80,
        'det_f1': 0.70,   'geom_ok_rate': 0.90,
    }
    logger.info('=' * 65)
    logger.info(f'批量评估结果（n={n}）')
    logger.info(f'{"指标":<20} {"均值":>8} {"std":>7} {"p50":>7}')
    for m, s in summary.items():
        target = paper_targets.get(m)
        flag   = (' ✓' if s['mean'] >= target else ' ✗') if target else ''
        logger.info(f'{m:<20} {s["mean"]:>8.4f} {s["std"]:>7.4f} {s["p50"]:>7.4f}{flag}')
    logger.info('=' * 65)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'batch_validation.json')
        with open(path, 'w') as f:
            json.dump({'summary': summary, 'n_valid': n, 'n_skip': skip}, f, indent=2)
        logger.info(f'批量验证结果已保存: {path}')

    return {'summary': summary, 'records': records}


def analyze_failures(
    records:    list,
    n_worst:    int  = 5,
    metric:     str  = 'iou_mask',
    output_dir: Optional[str] = None,
    data_folder: Optional[str] = None,
    cubicasa_root: Optional[str] = None,
):
    """找出指定指标最差的 n 个样本，保存可视化差异图"""
    if not records:
        logger.warning('无记录可分析')
        return

    worst = sorted(
        [r for r in records if metric in r],
        key=lambda x: x[metric],
    )[:n_worst]

    logger.info(f'最差 {n_worst} 个样本（按 {metric} 升序）:')
    for r in worst:
        logger.info(f'  {r["folder"]}  {metric}={r[metric]:.4f}')

    # 重新跑验证以获取中间结果（用于保存 diff 图）
    if output_dir and data_folder and cubicasa_root:
        os.makedirs(output_dir, exist_ok=True)
        for i, r in enumerate(worst[:3]):
            full_r = validate_single_sample(
                r['folder'], data_folder, cubicasa_root
            )
            if full_r.get('status') != 'ok':
                continue
            _save_diff_image(
                full_r,
                os.path.join(output_dir, f'failure_{i+1}_{r["folder"].replace("/", "_")}.png'),
            )


def _save_diff_image(result: dict, out_path: str):
    """保存 GT vs Pred 的差异图（TP=黄 FP=红 FN=蓝）"""
    gt   = result['_gt_crop']
    pred = result['_wall_pred']
    diff = np.zeros((*gt.shape, 3), dtype=np.uint8)
    diff[(pred == 1) & (gt == 1)] = [255, 255, 0]   # 黄 TP
    diff[(pred == 1) & (gt == 0)] = [255, 0,   0]   # 红 FP
    diff[(pred == 0) & (gt == 1)] = [0,   0, 255]   # 蓝 FN
    cv2.imwrite(out_path, cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))
    logger.info(f'差异图已保存: {out_path}')


# ══════════════════════════════════════════════════════════════
# 内部工具（验证系统使用）
# ══════════════════════════════════════════════════════════════

def _crop_to_wall(img_rgb, gt_wall, padding=10, min_size=64):
    h, w = img_rgb.shape[:2]
    if gt_wall.sum() == 0:
        return None, None
    rows = np.where(gt_wall.any(axis=1))[0]
    cols = np.where(gt_wall.any(axis=0))[0]
    y1 = max(rows[0]  - padding, 0)
    y2 = min(rows[-1] + padding + 1, h)
    x1 = max(cols[0]  - padding, 0)
    x2 = min(cols[-1] + padding + 1, w)
    if (y2 - y1) < min_size or (x2 - x1) < min_size:
        return None, None
    return img_rgb[y1:y2, x1:x2], gt_wall[y1:y2, x1:x2]


def _iou_mask(pred, gt):
    inter = (pred.astype(np.uint8) & gt.astype(np.uint8)).sum()
    union = (pred.astype(np.uint8) | gt.astype(np.uint8)).sum()
    return float(inter) / (float(union) + 1e-8)


def _det_metrics(pred_boxes, pred_scores, pred_labels,
                  gt_boxes, gt_labels,
                  iou_thresh=0.5, score_thresh=0.5):
    from torchvision.ops import box_iou as tv_box_iou
    keep  = pred_scores >= score_thresh
    pb    = pred_boxes[keep]
    tp = fp = fn = 0
    if len(gt_boxes) == 0:
        return len(pb), 0, 0
    if len(pb) == 0:
        return 0, 0, len(gt_boxes)
    iou_mat  = tv_box_iou(torch.tensor(pb, dtype=torch.float32),
                           torch.tensor(gt_boxes, dtype=torch.float32)).numpy()
    matched = set()
    for i in range(len(pb)):
        j = iou_mat[i].argmax()
        if iou_mat[i, j] >= iou_thresh and j not in matched:
            tp += 1; matched.add(j)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched)
    return tp, fp, fn


def _match_rate(det_boxes, det_labels, wall_mask, overlap_thresh=0.3):
    h, w = wall_mask.shape
    door_m = door_t = win_m = win_t = 0
    for box, label in zip(det_boxes, det_labels):
        x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
        x2 = min(w, int(box[2])); y2 = min(h, int(box[3]))
        if x2 <= x1 or y2 <= y1:
            continue
        area    = (x2 - x1) * (y2 - y1)
        overlap = wall_mask[y1:y2, x1:x2].sum() / (area + 1e-8)
        matched = overlap >= overlap_thresh
        if int(label) == 1:   door_t += 1; door_m += int(matched)
        elif int(label) == 2: win_t  += 1; win_m  += int(matched)
    return door_m, door_t, win_m, win_t


# ══════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Floor Plan Post-Processing Service')
    p.add_argument('--image',       default=None,  help='输入图片路径（单张后处理）')
    p.add_argument('--output_dir',  default=OUTPUT_DIR)
    p.add_argument('--no_3d',       action='store_true', help='跳过 3D 重建')
    p.add_argument('--validate',    action='store_true', help='运行验证模式')
    p.add_argument('--n_samples',   type=int, default=20, help='验证样本数')
    p.add_argument('--data_folder', default=None)
    p.add_argument('--cubicasa_root', default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.validate:
        from config import DATA_FOLDER, CUBICASA_ROOT
        data_folder    = args.data_folder or DATA_FOLDER
        cubicasa_root  = args.cubicasa_root or CUBICASA_ROOT
        result = batch_validate(
            data_folder   = data_folder,
            cubicasa_root = cubicasa_root,
            n_samples     = args.n_samples,
            save_dir      = args.output_dir,
        )
        analyze_failures(
            records       = result['records'],
            output_dir    = args.output_dir,
            data_folder   = data_folder,
            cubicasa_root = cubicasa_root,
        )

    elif args.image:
        result = run_postprocess(
            image_path = args.image,
            output_dir = args.output_dir,
            export_3d  = not args.no_3d,
        )
        logger.info(f'结果: walls={result["stats"]["n_walls"]}  '
                    f'doors={result["stats"]["n_doors"]}  '
                    f'windows={result["stats"]["n_windows"]}  '
                    f'elapsed={result["stats"]["elapsed"]}s')
    else:
        logger.error('请指定 --image 或 --validate')
        sys.exit(1)


if __name__ == '__main__':
    main()
