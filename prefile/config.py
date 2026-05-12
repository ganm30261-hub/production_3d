"""
config.py — 训练配置
所有超参数、路径、阈值集中在这里
修改训练策略只改这一个文件
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# ── 环境自动判断 ──
def _detect_paths():
    if os.path.exists('/workspace/production_3d'):
        return (
            '/workspace/CubiCasa5k',
            '/workspace/data/cubicasa5k/',
            '/workspace/production_3d/checkpoints_paper',
            '/workspace/production_3d/mlruns',
        )
    elif os.path.exists('/content/CubiCasa5k'):
        return (
            '/content/CubiCasa5k',
            '/content/data/cubicasa5k/',
            '/content/checkpoints_paper',
            '/content/mlruns',
        )
    else:
        return (
            r'E:\JOB\CubiCasa5k',
            r'C:/Users/kawayi_yaling/.cache/kagglehub/datasets/qmarva/cubicasa5k/versions/4/cubicasa5k/cubicasa5k/',
            './checkpoints_paper',
            './mlruns',
        )


CUBICASA_ROOT, DATA_FOLDER, CHECKPOINT_DIR, MLFLOW_DIR = _detect_paths()

# ── GCS ──
GCS_BUCKET       = 'yalingdata'
GCS_PREPROCESSED = 'preprocessed/cubicasa5k'
LOCAL_CACHE_DIR  = '/tmp/floorplan_cache'


@dataclass
class PaperConfig:
    # ── 数据 ──
    data_folder: str = DATA_FOLDER
    train_file:  str = 'train.txt'
    val_file:    str = 'val.txt'

    # ── 类别定义（CubiCasa5k）──
    wall_class_id:   int = 2   # seg[0] 中墙体的 class id
    seg_num_classes: int = 2   # background=0, wall=1

    # 门窗检测（Faster R-CNN）
    # 0=background, 1=door, 2=window
    det_num_classes: int = 3
    door_class_id:   int = 2   # icons seg[1] 中门的 class id
    window_class_id: int = 1   # icons seg[1] 中窗的 class id
    min_bbox_area:   int = 100 # 最小 bbox 面积（像素），过滤噪声

    # ── 论文预处理（Section 3）──
    crop_to_wall:  bool = True
    crop_padding:  int  = 10
    min_crop_size: int  = 64

    # ── 滑动窗口（Section 2.1）──
    tile_size:    int = 512
    tile_overlap: int = 64

    # ── 数据增强（Section 2.1）──
    aug_scale_range:      Tuple[float, float] = (0.5, 2.0)
    aug_rotation_degrees: List[int] = field(default_factory=lambda: [0, 90, 180, 270])
    aug_use_flip: bool = True

    # ── 归一化 ──
    norm_mean: Tuple = (0.485, 0.456, 0.406)
    norm_std:  Tuple = (0.229, 0.224, 0.225)

    # ── 模型（Section 2.2 + 2.3）──
    backbone:            str  = 'resnet50'
    pretrained_backbone: bool = True
    fpn_out_channels:    int  = 256

    # ── 训练超参数 ──
    batch_size:    int   = 4
    num_workers:   int   = 2
    max_epochs:    int   = 50
    learning_rate: float = 1e-4
    weight_decay:  float = 1e-4
    warmup_epochs: int   = 3
    use_amp:       bool  = True

    # ── 损失函数（Section 2.3）──
    use_affinity_loss:     bool  = True
    affinity_neighborhood: int   = 5
    use_kendall_weighting: bool  = True
    wall_class_weight:     float = 5.0

    # ── 检查点 ──
    checkpoint_dir: str = CHECKPOINT_DIR
    save_top_k:     int = 3

    # ── MLflow ──
    mlflow_experiment: str = 'floorplan_paper_method'
    mlflow_run_name:   str = f'paper_{time.strftime("%Y%m%d_%H%M%S")}'

    # ── GCS ──
    gcs_bucket:       str  = 'yalingdata'
    gcs_model_dir:    str  = 'models/paper_method/v1.0'
    gcs_preprocessed: str  = 'preprocessed/cubicasa5k'
    use_preprocessed: bool = True

    # ── 图片类型自适应预处理 ──
    # 'auto' | 'cad' | 'phone' | 'scan' | 'bw'
    image_type:         str  = 'auto'
    preprocess_enabled: bool = True

    def validate(self):
        if not self.data_folder.startswith('gs://'):
            assert Path(self.data_folder).exists(), \
                f'数据目录不存在: {self.data_folder}'
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
