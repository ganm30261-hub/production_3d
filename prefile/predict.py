"""
predict.py — 推理主程序

加载训练好的 best_model.pth，对单张图片进行：
  1. 墙体分割（输出二值 mask）
  2. 门窗检测（输出 bounding boxes）

用法：
  python predict.py --image path/to/floor.png --checkpoint path/to/best_model.pth
  python predict.py --image floor.png   # 自动查找 CHECKPOINT_DIR/best_model.pth
"""

import argparse
import json
import logging
import os
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms

from config import PaperConfig, CUBICASA_ROOT, CHECKPOINT_DIR
from model_arch import FloorplanModel, build_model
from utils import adaptive_preprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, cfg: PaperConfig, device: str) -> FloorplanModel:
    """从 .pth 文件加载模型权重"""
    model = build_model(cfg, device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    epoch    = ckpt.get('epoch', '?')
    wall_iou = ckpt.get('val_wall_iou', '?')
    det_f1   = ckpt.get('det_f1', '?')
    logger.info(f'模型加载成功  epoch={epoch}  val_wall_iou={wall_iou}  det_f1={det_f1}')
    return model


# ══════════════════════════════════════════════════════════════
# 单张图片推理
# ══════════════════════════════════════════════════════════════

def preprocess_image(
    image_path: str,
    cfg: PaperConfig,
    tile_size: int = 512,
) -> torch.Tensor:
    """读取图片 → 自适应预处理 → 归一化 → tensor"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f'图像不存在: {image_path}')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 自适应预处理
    img_rgb = adaptive_preprocess(img_rgb, cfg.image_type)

    # Resize 到 tile_size（推理时整图输入）
    img_resized = cv2.resize(img_rgb, (tile_size, tile_size),
                             interpolation=cv2.INTER_LINEAR)

    img_t = transforms.ToTensor()(img_resized)
    img_t = transforms.Normalize(cfg.norm_mean, cfg.norm_std)(img_t)
    return img_t.unsqueeze(0)   # (1, 3, H, W)


def predict_single(
    image_path:      str,
    model:           FloorplanModel,
    cfg:             PaperConfig,
    device:          str,
    score_threshold: float = 0.5,
) -> dict:
    """
    对单张图片执行推理。
    返回：
      wall_mask   : np.ndarray (H, W) uint8，1=墙体
      boxes       : np.ndarray (N, 4) float32，[x1,y1,x2,y2]
      labels      : np.ndarray (N,) int，1=door 2=window
      scores      : np.ndarray (N,) float，置信度
    """
    img_tensor = preprocess_image(image_path, cfg, cfg.tile_size).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    # ── 墙体分割 ──
    seg_logits = outputs['seg_logits']          # (1, num_classes, H, W)
    pred_mask  = seg_logits.argmax(dim=1)[0]    # (H, W)
    wall_mask  = pred_mask.cpu().numpy().astype(np.uint8)

    # ── 门窗检测 ──
    det = outputs['det_outputs'][0]
    boxes  = det['boxes'].cpu().numpy()
    labels = det['labels'].cpu().numpy()
    scores = det['scores'].cpu().numpy()

    # 过滤低置信度
    keep   = scores >= score_threshold
    boxes  = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    return {
        'wall_mask': wall_mask,
        'boxes':     boxes,
        'labels':    labels,
        'scores':    scores,
    }


# ══════════════════════════════════════════════════════════════
# 结果可视化（保存图片，不依赖 display）
# ══════════════════════════════════════════════════════════════

def save_result(
    image_path: str,
    result:     dict,
    output_path: str,
    tile_size:  int = 512,
):
    """把墙体 mask 和门窗 bbox 叠加到原图，保存为文件"""
    img_bgr     = cv2.imread(image_path)
    img_resized = cv2.resize(img_bgr, (tile_size, tile_size))

    # 墙体 mask 叠加（红色半透明）
    overlay    = img_resized.copy()
    wall_layer = np.zeros_like(overlay)
    wall_layer[result['wall_mask'] == 1] = [0, 0, 200]
    cv2.addWeighted(wall_layer, 0.4, overlay, 0.6, 0, overlay)

    # 门窗 bbox
    label_names = {1: 'Door', 2: 'Window'}
    colors      = {1: (255, 100, 0), 2: (0, 100, 255)}   # BGR

    for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
        x1, y1, x2, y2 = map(int, box)
        color = colors.get(label, (200, 200, 0))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        text = f'{label_names.get(label, "?")}: {score:.2f}'
        cv2.putText(overlay, text, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.imwrite(output_path, overlay)
    logger.info(f'结果已保存: {output_path}')


# ══════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Floor Plan Inference')
    p.add_argument('--image',      required=True,  help='输入图片路径')
    p.add_argument('--checkpoint', default=None,   help='模型权重路径（默认 CHECKPOINT_DIR/best_model.pth）')
    p.add_argument('--output',     default=None,   help='结果图片保存路径（默认与输入同目录）')
    p.add_argument('--score_thr',  type=float, default=0.5,  help='检测置信度阈值')
    p.add_argument('--json',       action='store_true',       help='同时输出 JSON 格式的检测结果')
    return p.parse_args()


def main():
    args = parse_args()

    os.chdir(CUBICASA_ROOT)
    sys.path.insert(0, CUBICASA_ROOT)

    cfg    = PaperConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 确定 checkpoint 路径
    ckpt_path = args.checkpoint or os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        logger.error(f'Checkpoint 不存在: {ckpt_path}')
        sys.exit(1)

    model = load_model(ckpt_path, cfg, device)

    # 推理
    result = predict_single(args.image, model, cfg, device, args.score_thr)

    n_det = len(result['boxes'])
    wall_pct = result['wall_mask'].mean() * 100
    logger.info(f'wall pixel %={wall_pct:.1f}%  检测到 {n_det} 个门窗')

    # 保存可视化结果
    output_path = args.output or args.image.replace('.png', '_pred.png').replace('.jpg', '_pred.jpg')
    save_result(args.image, result, output_path, cfg.tile_size)

    # 可选：输出 JSON
    if args.json:
        json_path = output_path.rsplit('.', 1)[0] + '.json'
        payload   = {
            'image':    args.image,
            'wall_pct': round(float(wall_pct), 2),
            'detections': [
                {
                    'box':   result['boxes'][i].tolist(),
                    'label': int(result['labels'][i]),
                    'score': round(float(result['scores'][i]), 4),
                }
                for i in range(n_det)
            ],
        }
        with open(json_path, 'w') as f:
            json.dump(payload, f, indent=2)
        logger.info(f'JSON 结果已保存: {json_path}')


if __name__ == '__main__':
    main()
