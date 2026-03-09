# %% [markdown]
# # CubiCasa5k 平面图识别 — 交互式训练
#
# 本 notebook 将数据导出和三个模型训练串联为完整流程。
# 每个 cell 可独立执行，方便调试和观察中间结果。
#
# **三个模型:**
# 1. 墙体分割 (FPN + ResNet) — 论文 Section 2.3
# 2. 门窗检测 (Faster R-CNN) — 论文 Section 2.2
# 3. YOLO 实例分割 — 原有方案

# %% [markdown]
# ## 0. 环境配置

# %%
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('notebook')

# %%
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    GPU_IDS = list(range(torch.cuda.device_count()))
    DEVICE = f'cuda:{GPU_IDS[0]}'
else:
    GPU_IDS = []
    DEVICE = 'cpu'
print(f"Using: {DEVICE}, GPU IDs: {GPU_IDS}")

# %% [markdown]
# ## 1. 配置

# %%
# ===== 根据你的环境修改以下路径 =====
DATA_ROOT = '/root/.cache/kagglehub/datasets/qmarva/cubicasa5k/versions/4/cubicasa5k/cubicasa5k'
OUTPUT_DIR = './output'
SAVE_DIR = './runs'
SPLITS = ['train', 'val']

# 墙体分割配置
WALL_SEG_CONFIG = {
    'backbone': 'resnet50',
    'epochs': 50,
    'batch_size': 4 * max(len(GPU_IDS), 1),   # 多 GPU 自动扩展
    'lr': 1e-4,
    'target_size': 512,
}

# 门窗检测配置
SYMBOL_DET_CONFIG = {
    'epochs': 15,
    'batch_size': 2 * max(len(GPU_IDS), 1),
    'lr': 5e-3,
}

# YOLO 配置
YOLO_CONFIG = {
    'model': 'yolov8m-seg.pt',
    'epochs': 100,
    'batch_size': 8,
    'imgsz': 640,
}

# %% [markdown]
# ## 2. 数据导出
#
# 从 CubiCasa5k 的 SVG 标注导出三种格式的训练数据。
# 已导出过的可以跳过这个 section。

# %%
from b_data.pipeline import FloorplanPipeline, PipelineConfig

data_pipeline = FloorplanPipeline(PipelineConfig(
    data_root=DATA_ROOT,
    wall_seg_output_dir=f'{OUTPUT_DIR}/wall_segmentation',
    symbol_det_output_dir=f'{OUTPUT_DIR}/symbol_detection',
    yolo_output_dir=f'{OUTPUT_DIR}/yolo_dataset',
    coco_output_dir=f'{OUTPUT_DIR}/coco_annotations',
))

# %% [markdown]
# ### 2a. 导出墙体分割数据 (images/ + masks/)

# %%
wall_seg_export = data_pipeline.export_wall_segmentation_data(
    splits=SPLITS, crop_to_walls=True,
)
print("墙体分割导出结果:", wall_seg_export)

# %% [markdown]
# ### 2b. 导出门窗检测数据 (images/ + annotations.json)

# %%
symbol_det_export = data_pipeline.export_symbol_detection_data(
    splits=SPLITS, crop_to_walls=True,
)
print("门窗检测导出结果:", symbol_det_export)

# %% [markdown]
# ### 2c. 导出 YOLO 数据 (dataset.yaml)

# %%
yolo_export = data_pipeline.run_full(splits=SPLITS, target_format='yolo')
print("YOLO 导出完成")

# %% [markdown]
# ### 2d. 数据导出验证

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

def verify_wall_seg_data(data_dir, split='train', n=3):
    """可视化墙体分割数据样本。"""
    img_dir = Path(data_dir) / split / 'images'
    msk_dir = Path(data_dir) / split / 'masks'
    files = sorted(img_dir.glob('*.png'))[:n]

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(msk_dir / f.name), cv2.IMREAD_GRAYSCALE)

        overlay = img.copy()
        overlay[mask > 127] = [0, 255, 0]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Image: {f.name}')
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'Wall Mask')
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f'Overlay')

    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

verify_wall_seg_data(f'{OUTPUT_DIR}/wall_segmentation')

# %%
import json

def verify_symbol_det_data(data_dir, split='train', n=3):
    """可视化门窗检测数据样本。"""
    img_dir = Path(data_dir) / split / 'images'
    with open(Path(data_dir) / split / 'annotations.json') as f:
        anns = json.load(f)

    fig, axes = plt.subplots(1, min(n, len(anns)), figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    colors = {1: (0, 255, 0), 2: (0, 0, 255)}  # window=green, door=blue
    names = {1: 'window', 2: 'door'}

    for i, ann in enumerate(anns[:n]):
        img = cv2.imread(str(img_dir / ann['file_name']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for box, label in zip(ann['boxes'], ann['labels']):
            x1, y1, x2, y2 = [int(v) for v in box]
            color = colors.get(label, (255, 0, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, names.get(label, '?'), (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        axes[i].imshow(img)
        axes[i].set_title(f'{ann["file_name"]}: {len(ann["boxes"])} objects')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

verify_symbol_det_data(f'{OUTPUT_DIR}/symbol_detection')

# %% [markdown]
# ## 3. 训练模型
#
# 三个模型依次训练。每个模型的训练器是独立的，可以单独跑。

# %% [markdown]
# ### 3a. 墙体分割 (FPN + ResNet + Affinity + Kendall)

# %%
from c_models.wall_seg_trainer import WallSegmentationTrainer, WallSegTrainerConfig

wall_seg_cfg = WallSegTrainerConfig(
    data_dir=f'{OUTPUT_DIR}/wall_segmentation',
    backbone=WALL_SEG_CONFIG['backbone'],
    pretrained=True,
    target_size=WALL_SEG_CONFIG['target_size'],
    num_epochs=WALL_SEG_CONFIG['epochs'],
    batch_size=WALL_SEG_CONFIG['batch_size'],
    lr=WALL_SEG_CONFIG['lr'],
    num_workers=4,
    device=DEVICE,
    use_affinity=True,
    use_kendall=True,
    save_dir=f'{SAVE_DIR}/wall_segmentation',
)

wall_seg_trainer = WallSegmentationTrainer(wall_seg_cfg)
if len(GPU_IDS) > 1:
    wall_seg_trainer._multi_gpu_ids = GPU_IDS

wall_seg_results = wall_seg_trainer.train()
print(f"墙体分割完成, Best IoU: {wall_seg_results['best_val_iou']:.4f}")

# %% [markdown]
# ### 3b. 门窗检测 (Faster R-CNN)

# %%
from c_models.symbol_det_trainer import SymbolDetectionTrainer, SymbolDetTrainerConfig

symbol_det_cfg = SymbolDetTrainerConfig(
    data_dir=f'{OUTPUT_DIR}/symbol_detection',
    pretrained=True,
    num_epochs=SYMBOL_DET_CONFIG['epochs'],
    batch_size=SYMBOL_DET_CONFIG['batch_size'],
    lr=SYMBOL_DET_CONFIG['lr'],
    num_workers=4,
    device=DEVICE,
    save_dir=f'{SAVE_DIR}/symbol_detection',
)

symbol_det_trainer = SymbolDetectionTrainer(symbol_det_cfg)
if len(GPU_IDS) > 1:
    symbol_det_trainer._multi_gpu_ids = GPU_IDS

symbol_det_results = symbol_det_trainer.train()
print(f"门窗检测完成, Best Loss: {symbol_det_results['best_val_loss']:.4f}")

# %% [markdown]
# ### 3c. YOLO 实例分割

# %%
from c_models.yolo_trainer import EnhancedYOLOTrainer, YOLOTrainerConfig

yolo_device = ','.join(str(g) for g in GPU_IDS) if GPU_IDS else 'cpu'

yolo_cfg = YOLOTrainerConfig(
    model_name=YOLO_CONFIG['model'],
    epochs=YOLO_CONFIG['epochs'],
    batch_size=YOLO_CONFIG['batch_size'],
    imgsz=YOLO_CONFIG['imgsz'],
    device=yolo_device,
    save_dir=f'{SAVE_DIR}/yolo',
)

yolo_trainer = EnhancedYOLOTrainer(yolo_cfg)
yolo_results = yolo_trainer.train(
    data_yaml=f'{OUTPUT_DIR}/yolo_dataset/dataset.yaml',
)
print("YOLO 训练完成")

# %% [markdown]
# ## 4. 训练结果可视化

# %%
def plot_training_history(history, title, metric_key, ylabel):
    """绘制训练曲线。"""
    epochs = [h['epoch'] for h in history]
    values = [h[metric_key] for h in history]
    losses = [h['train_loss'] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, losses, 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} — Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, values, 'r-', label=ylabel)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(ylabel)
    ax2.set_title(f'{title} — {ylabel}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 墙体分割训练曲线
if 'history' in wall_seg_results:
    plot_training_history(
        wall_seg_results['history'],
        '墙体分割 (FPN)', 'val_iou', 'Val IoU',
    )

# 门窗检测训练曲线
if 'history' in symbol_det_results:
    plot_training_history(
        symbol_det_results['history'],
        '门窗检测 (Faster R-CNN)', 'val_loss', 'Val Loss',
    )

# %% [markdown]
# ## 5. 结果总结

# %%
print("=" * 60)
print("训练结果总结")
print("=" * 60)
print(f"墙体分割 (FPN):       Best Val IoU  = {wall_seg_results.get('best_val_iou', 'N/A')}")
print(f"门窗检测 (Faster RCNN): Best Val Loss = {symbol_det_results.get('best_val_loss', 'N/A')}")
print(f"YOLO:                  见 {SAVE_DIR}/yolo/ 下的 results.csv")
print("=" * 60)
print(f"\n模型权重保存在:")
print(f"  墙体分割: {SAVE_DIR}/wall_segmentation/best.pth")
print(f"  门窗检测: {SAVE_DIR}/symbol_detection/best.pth")
print(f"  YOLO:     {SAVE_DIR}/yolo/enhanced_training/weights/best.pt")
