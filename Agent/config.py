# config.py
import os
import sys
import logging
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import torch

# ══════════════════════════════════════════════════════════════
# 路径（自动识别运行环境）
# ══════════════════════════════════════════════════════════════

if os.path.exists('/workspace/production_3d'):
    PROJECT_ROOT  = '/workspace/production_3d'
    CUBICASA_ROOT = '/workspace/CubiCasa5k'
    DATA_FOLDER   = '/workspace/data/cubicasa5k/'
    COMPANY_DIR   = '/workspace/data/company_floorplans/'
    SAM2_CKPT     = '/workspace/checkpoints/sam2_hiera_large.pt'
elif os.path.exists('/content'):
    PROJECT_ROOT  = '/content/production_3d'
    CUBICASA_ROOT = '/content/CubiCasa5k'
    DATA_FOLDER   = '/content/data/cubicasa5k/'
    COMPANY_DIR   = '/content/data/company_floorplans/'
    SAM2_CKPT     = '/content/checkpoints/sam2_hiera_large.pt'
else:
    PROJECT_ROOT  = '.'
    CUBICASA_ROOT = r'E:\JOB\CubiCasa5k'
    DATA_FOLDER   = r'C:/Users/kawayi_yaling/.cache/kagglehub/datasets/qmarva/cubicasa5k/versions/4/cubicasa5k/cubicasa5k/'
    COMPANY_DIR   = './data/company_floorplans/'
    SAM2_CKPT     = './checkpoints/sam2_hiera_large.pt'

CKPT_DIR       = os.path.join(PROJECT_ROOT, 'checkpoints_dinov2_lora')
PSEUDO_OUT_DIR = os.path.join(PROJECT_ROOT, 'pseudo_labels')
MLFLOW_DIR     = os.path.join(PROJECT_ROOT, 'mlruns')

for d in [CKPT_DIR, PSEUDO_OUT_DIR, MLFLOW_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CUBICASA_ROOT)
sys.path.insert(0, '/workspace/production_3d/Agent/postfile')

# ══════════════════════════════════════════════════════════════
# Logger + Device（全局单例，其他模块从这里 import）
# ══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ══════════════════════════════════════════════════════════════
# split 文件映射（被 PseudoLabelConfig 的 property 调用）
# ══════════════════════════════════════════════════════════════

def _version_to_files(version: str, split: str) -> List[str]:
    mapping = {
        'hq':       [f'{split}_hq.txt'],
        'hq_arch':  [f'{split}_hq_arch.txt'],
        'combined': [f'{split}_hq.txt', f'{split}_hq_arch.txt'],
    }
    if version not in mapping:
        raise ValueError(
            f'dataset_version 必须是 hq/hq_arch/combined，收到: {version}'
        )
    return mapping[version]


# ══════════════════════════════════════════════════════════════
# PseudoLabelConfig
# ══════════════════════════════════════════════════════════════

@dataclass
class PseudoLabelConfig:
    # ── 数据集版本（三选一）──
    # 'hq'       → 只用 high_quality
    # 'hq_arch'  → 只用 high_quality_architectural
    # 'combined' → 两者合集（推荐）
    dataset_version: str = 'combined'

    # ── DINOv2 + LoRA ──
    dinov2_model:     str       = 'vit_large_patch14_dinov2'
    lora_r:           int       = 16
    lora_alpha:       int       = 32
    lora_dropout:     float     = 0.1
    lora_target:      List[str] = field(default_factory=lambda: ['qkv', 'proj'])
    fpn_out_channels: int       = 256
    seg_num_classes:  int       = 2    # background=0, wall=1
    det_num_classes:  int       = 3    # background=0, door=1, window=2

    # ── 数据路径 ──
    data_folder: str = DATA_FOLDER

    # ── 类别 ID（CubiCasa5k 约定）──
    wall_class_id:   int = 2
    door_class_id:   int = 2
    window_class_id: int = 1
    min_bbox_area:   int = 100

    # ── 图像处理 ──
    tile_size:    int   = 518   # DINOv2 patch_size=14，必须是14的倍数 (37×14=518)
    tile_overlap: int   = 56    # 14的倍数，stride=462=33×14
    norm_mean:    Tuple = (0.485, 0.456, 0.406)
    norm_std:     Tuple = (0.229, 0.224, 0.225)

    # ── 训练超参 ──
    batch_size:        int   = 4
    num_workers:       int   = 2
    max_epochs:        int   = 50
    learning_rate:     float = 5e-5
    weight_decay:      float = 1e-4
    warmup_epochs:     int   = 3
    use_amp:           bool  = True
    grad_clip:         float = 3.0
    wall_class_weight: float = 5.0
    dice_weight:       float = 0.5
    bce_weight:        float = 0.5

    # ── 断点续训 ──
    # 填入要恢复的 checkpoint 路径，None = 从头开始
    # 例: resume_from = '/workspace/.../hq/hq_bs4_lr5e-5_ep020_iou0.7823.pth'
    resume_from: str = None

    # ── GCS 配置 ──
    gcs_bucket:  str = 'yalingdata'
    gcs_project: str = 'project-d3027d52-508f-4689-899'

    # ── Step 2: SAM2 精化 ──
    sam2_ckpt:         str   = SAM2_CKPT
    sam2_cfg:          str   = 'sam2_hiera_l.yaml'
    sam2_n_pos_points: int   = 5
    sam2_n_neg_points: int   = 3
    sam2_score_thresh: float = 0.8

    # ── Step 3: VLM 语义补全 ──
    vlm_model:      str = 'claude-sonnet-4-20250514'
    vlm_max_tokens: int = 1024

    # ── Step 4: SVG 生成 ──
    pixels_per_meter:  float = 50.0
    wall_height_m:     float = 2.8
    door_height_m:     float = 2.1
    window_height_m:   float = 1.2
    shrink_iou_thresh: float = 0.85
    min_segment_area:  int   = 200

    # ── 输出 ──
    pseudo_out_dir: str = PSEUDO_OUT_DIR
    company_dir:    str = COMPANY_DIR

    # ── 检查点目录：按 dataset_version 自动分子目录 ──
    @property
    def checkpoint_dir(self) -> str:
        d = os.path.join(CKPT_DIR, self.dataset_version)
        os.makedirs(d, exist_ok=True)
        return d

    # ── MLflow ──
    @property
    def mlflow_experiment(self) -> str:
        return 'dinov2_lora_floorplan'

    @property
    def mlflow_run_name(self) -> str:
        lr_str = f'{self.learning_rate:.0e}'.replace('-0', '-')
        return (
            f'{self.dataset_version}'
            f'_ep{self.max_epochs}'
            f'_bs{self.batch_size}'
            f'_lr{lr_str}'
            f'_{time.strftime("%m%d_%H%M")}'
        )

    # ── split 文件（由 dataset_version 自动推导）──
    @property
    def train_files(self) -> List[str]:
        return _version_to_files(self.dataset_version, 'train')

    @property
    def val_files(self) -> List[str]:
        return _version_to_files(self.dataset_version, 'val')

    # ── GCS 模型目录 ──
    @property
    def gcs_model_dir(self) -> str:
        lr_str  = f'{self.learning_rate:.0e}'.replace('-0', '-')
        run_tag = (
            f'{self.dataset_version}'
            f'_ep{self.max_epochs}'
            f'_bs{self.batch_size}'
            f'_lr{lr_str}'
        )
        return f'models/dinov2_lora/{self.dataset_version}/{run_tag}'


# ══════════════════════════════════════════════════════════════
# 全局默认配置（其他模块 from config import CFG 直接用）
# ══════════════════════════════════════════════════════════════

CFG = PseudoLabelConfig()
