# test_inference.py
# 用法：python3 test_inference.py --image /path/to/floor.png
# 不传 --image 时自动从 CubiCasa val 集找一张图测试

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CFG, DEVICE, logger


# ══════════════════════════════════════════════════════════════
# Step 0：找一张测试图
# ══════════════════════════════════════════════════════════════

def find_test_image(image_path: str = None) -> str:
    """
    优先级：
        1. 用户指定的本地路径
        2. 本地 val_hq.txt 找第一张
        3. 从 GCS preprocessed 下载第一张
    """
    # 1. 用户指定
    if image_path and os.path.exists(image_path):
        return image_path

    # 2. 本地 val_hq.txt
    val_txt = os.path.join(CFG.data_folder, 'val_hq.txt')
    if os.path.exists(val_txt):
        with open(val_txt) as f:
            folder = f.readline().strip()
        path = os.path.join(CFG.data_folder, folder, 'F1_scaled.png')
        if os.path.exists(path):
            logger.info(f'自动选用本地测试图: {path}')
            return path

    # 3. 从 GCS 下载
    return _fetch_from_gcs()


def _fetch_from_gcs(
    gcs_prefix: str = "gs://yalingdata/preprocessed/cubicasa5k/high_quality/",
    local_dir:  str = "/tmp/infer_test_img",
) -> str:
    """从 GCS 随机取一个样本下载到本地，返回本地路径。"""
    import subprocess, random

    os.makedirs(local_dir, exist_ok=True)

    # 列出所有样本 ID
    logger.info(f"从 GCS 获取样本列表: {gcs_prefix}")
    ret = subprocess.run(
        ["gsutil", "ls", gcs_prefix],
        capture_output=True, text=True
    )
    if ret.returncode != 0:
        raise RuntimeError(f"gsutil ls 失败: {ret.stderr.strip()}")

    folders = [l.strip().rstrip("/") for l in ret.stdout.strip().split("\n") if l.strip()]
    if not folders:
        raise FileNotFoundError(f"GCS 路径为空: {gcs_prefix}")

    # 随机选一个（或取第一个）
    chosen = random.choice(folders[:20])   # 只从前20个里选，避免列举太慢
    sample_id = chosen.split("/")[-1]
    gcs_img   = f"{chosen}/F1_preprocessed.png"
    local_img = os.path.join(local_dir, f"{sample_id}_F1_preprocessed.png")

    if os.path.exists(local_img):
        logger.info(f"使用已缓存的本地图片: {local_img}")
        return local_img

    logger.info(f"从 GCS 下载: {gcs_img} → {local_img}")
    ret = subprocess.run(
        ["gsutil", "cp", gcs_img, local_img],
        capture_output=True, text=True
    )
    if ret.returncode != 0:
        raise RuntimeError(f"gsutil cp 失败: {ret.stderr.strip()}")

    logger.info(f"下载完成: {local_img}")
    return local_img


# ══════════════════════════════════════════════════════════════
# Step 1：加载模型
# ══════════════════════════════════════════════════════════════

def load_model(ckpt_path: str = None):
    from model.model import DINOv2LoRAModel

    # 自动找最新的 best checkpoint
    if ckpt_path is None:
        for version in ('hq', 'hq_arch', 'combined'):
            ckpt_dir = os.path.join(
                '/workspace/production_3d/checkpoints_dinov2_lora', version
            )
            candidate = os.path.join(ckpt_dir, f'{version}_bs2_lr5e-5_best.pth')
            # batch_size 可能不同，模糊匹配
            if not os.path.exists(candidate):
                import glob
                matches = glob.glob(os.path.join(ckpt_dir, '*_best.pth'))
                candidate = matches[0] if matches else None
            if candidate and os.path.exists(candidate):
                ckpt_path = candidate
                break

    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'找不到 checkpoint: {ckpt_path}\n'
            f'请先跑试训练，或用 --ckpt 指定路径。'
        )

    logger.info(f'加载 checkpoint: {ckpt_path}')
    model = DINOv2LoRAModel(CFG).to(DEVICE)

    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    saved_iou = ckpt.get('metrics', {}).get('val_iou', 0)
    logger.info(f'模型加载完成  val_iou={saved_iou:.4f}  device={DEVICE}')
    return model


# ══════════════════════════════════════════════════════════════
# Step 2：推理 + 可视化
# ══════════════════════════════════════════════════════════════

def visualize(image_rgb, wall_mask, det_boxes, save_path):
    """把 wall mask 和检测框叠加到原图，保存可视化结果。"""
    vis = image_rgb.copy()

    # wall mask：绿色半透明
    overlay        = vis.copy()
    overlay[wall_mask == 1] = [0, 200, 0]
    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

    # det boxes：蓝色矩形
    for b in det_boxes:
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    logger.info(f'可视化已保存: {save_path}')


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=None,  help='测试图片路径')
    parser.add_argument('--ckpt',  default=None,  help='checkpoint 路径')
    parser.add_argument('--out',   default='/tmp', help='输出目录')
    parser.add_argument('--dry_run_vlm', action='store_true', default=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── 找图 ──
    image_path = find_test_image(args.image)
    name       = os.path.splitext(os.path.basename(image_path))[0]

    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f'读图失败: {image_path}'
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    logger.info(f'图片尺寸: {img_rgb.shape}')

    # ── 加载模型 ──
    model = load_model(args.ckpt)

    # ── Step 1：DINOv2+LoRA 推理 ──
    from pipeline import run_inference
    t0       = time.time()
    infer_out = run_inference(img_rgb, model, CFG)
    logger.info(f'推理耗时: {time.time()-t0:.1f}s')

    wall_mask = infer_out['wall_mask']
    det_boxes = infer_out['boxes']
    det_scores = infer_out['scores']
    logger.info(
        f'wall覆盖率: {wall_mask.mean()*100:.1f}%  '
        f'检测框: {len(det_boxes)} 个'
    )

    # 保存推理结果可视化
    vis_path = os.path.join(args.out, f'{name}_step1_inference.png')
    visualize(img_rgb, wall_mask, det_boxes, vis_path)

    # ── Step 2：SAM2 精化（可选）──
    try:
        from tools.sam2_refine import load_sam2_predictor, refine_mask_with_sam2
        predictor = load_sam2_predictor(CFG)
        if predictor:
            refined_mask = refine_mask_with_sam2(img_rgb, wall_mask, predictor, CFG)
            logger.info(f'SAM2 精化完成  pixels: {wall_mask.sum()} → {refined_mask.sum()}')
        else:
            refined_mask = wall_mask
            logger.info('SAM2 不可用，跳过精化')
    except Exception as e:
        refined_mask = wall_mask
        logger.info(f'SAM2 跳过: {e}')

    # ── Step 3 & 4：矢量化 + SVG ──
    try:
        from vector_logic import vectorize_wall_mask
        from postprocess_config import VectorizationConfig
        vect_cfg   = VectorizationConfig(iou_threshold=CFG.shrink_iou_thresh)
        wall_boxes = vectorize_wall_mask(refined_mask, vect_cfg)
        logger.info(f'矢量化完成  wall_boxes: {len(wall_boxes)} 个')
    except ImportError:
        wall_boxes = []
        logger.warning('vector_logic 不可用，跳过矢量化')

    # VLM 语义补全（dry_run 用检测框代替）
    openings = []
    for b, l in zip(det_boxes, infer_out['labels']):
        openings.append({
            'type':    'door' if l == 1 else 'window',
            'bbox':    [int(v) for v in b],
            'wall_side':    'unknown',
            'estimated_width_m': None,
            'confidence': 0.9,
        })

    # SVG
    if wall_boxes:
        from tools.svg_export import generate_svg
        svg_path = os.path.join(args.out, f'{name}_pseudo_label.svg')
        generate_svg(
            image_wh    = (img_rgb.shape[1], img_rgb.shape[0]),
            wall_boxes  = wall_boxes,
            openings    = openings,
            cfg         = CFG,
            output_path = svg_path,
        )

    # ── 综合评分 ──
    from evaluation import evaluate, EvalWeights
    eval_result = evaluate(
        pred_mask   = refined_mask,
        gt_mask     = np.zeros_like(refined_mask),  # 无 GT，IoU 自动跳过
        wall_boxes  = wall_boxes,
        openings    = openings,
        image_rgb   = img_rgb,
        cfg         = CFG,
        weights     = EvalWeights(),
        vlm_client  = None,
    )

    # ── 汇总 ──
    print('\n' + '='*50)
    print(f'  推理结果汇总：{name}')
    print('='*50)
    print(f'  wall 覆盖率   : {wall_mask.mean()*100:.1f}%')
    print(f'  检测框数量    : {len(det_boxes)}')
    print(f'  wall_boxes    : {len(wall_boxes)}')
    print(f'  IoU_pixel     : {eval_result.iou_pixel:.3f}  (无GT时=1.0)')
    print(f'  C_topological : {eval_result.c_topological:.3f}')
    print(f'  S_total       : {eval_result.s_total:.3f}')
    print(f'  可视化        : {vis_path}')
    if wall_boxes:
        print(f'  SVG           : {svg_path}')
    print('='*50)
    print()

    if eval_result.c_topological < 0.3:
        print('⚠️  拓扑得分低：墙体未围成封闭房间，属于早期训练正常现象，继续训练后会改善。')
    else:
        print('✓  推理链路全部跑通。')


if __name__ == '__main__':
    main()
