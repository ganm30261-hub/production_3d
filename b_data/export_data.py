"""
本地数据导出使用指南
===================

前提条件：
1. CubiCasa5k 数据集已下载到本地
2. floortrans 库已安装（用于解析 SVG）
3. Python 环境已装好 cv2, numpy, torch, tqdm 等依赖

你的项目目录结构应该是：
    project/
    ├── data/                          ← 你上传的 8 个文件放在这里
    │   ├── __init__.py
    │   ├── exceptions.py
    │   ├── schemas.py
    │   ├── cubicasa_parser.py
    │   ├── preprocessing.py
    │   ├── coco_converter.py
    │   ├── yolo_converter.py
    │   └── pipeline.py
    │
    ├── floortrans/                    ← CubiCasa5k 官方的 House 解析库
    │   └── loaders/
    │       └── house.py
    │
    ├── export_data.py                 ← 下面会创建的运行脚本
    │
    └── cubicasa5k/                    ← 数据集目录（或在其他路径）
        ├── train.txt (或 train_hq_arch.txt)
        ├── val.txt   (或 val_hq_arch.txt)
        ├── test.txt  (或 test_hq_arch.txt)
        └── high_quality_architectural/
            ├── 1/
            │   ├── F1_scaled.png
            │   ├── F1_original.png
            │   └── model.svg
            ├── 2/
            │   └── ...
            └── ...
"""


# ============================================================================
# 方法 1: 命令行直接执行（最简单）
# ============================================================================

"""
打开终端，cd 到你的 project/ 目录，然后运行：

# ---- 通道 1: 导出 YOLO 格式（原有方案）----
# 输出: output/coco_annotations/ + output/yolo_dataset/
python -m data.pipeline --data_root /你的/cubicasa5k/路径 --target yolo --splits train val

# ---- 通道 2: 导出论文方法的两套数据 ----
# 墙体分割数据 → output/wall_segmentation/train/images/ + masks/
# 门窗检测数据 → output/symbol_detection/train/images/ + annotations.json
python -m data.pipeline --data_root /你的/cubicasa5k/路径 --target all_paper --splits train val

# ---- 一次性全部导出 ----
# 先跑 YOLO，再跑论文方法
python -m data.pipeline --data_root /你的/cubicasa5k/路径 --target yolo --splits train val
python -m data.pipeline --data_root /你的/cubicasa5k/路径 --target all_paper --splits train val

# 查看导出结果
ls -la output/yolo_dataset/
ls -la output/wall_segmentation/train/images/ | head
ls -la output/wall_segmentation/train/masks/ | head
ls -la output/symbol_detection/train/
"""


# ============================================================================
# 方法 2: Python 脚本执行（推荐，可控制更多参数）
# ============================================================================

# 把以下代码保存为 project/export_data.py，然后运行 python export_data.py

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ===== 修改这里为你的实际路径 =====
DATA_ROOT = '/你的/cubicasa5k/路径'
# 例如:
# DATA_ROOT = r'C:\Users\你的用户名\cubicasa5k'                    # Windows
# DATA_ROOT = '/home/你的用户名/cubicasa5k'                        # Linux
# DATA_ROOT = '/root/.cache/kagglehub/datasets/qmarva/cubicasa5k/versions/4/cubicasa5k/cubicasa5k'  # Kaggle 下载

OUTPUT_DIR = './output'
SPLITS = ['train', 'val']

# ==================================

from b_data.pipeline import FloorplanPipeline, PipelineConfig

# 创建流水线
pipeline = FloorplanPipeline(PipelineConfig(
    data_root=DATA_ROOT,
    coco_output_dir=f'{OUTPUT_DIR}/coco_annotations',
    yolo_output_dir=f'{OUTPUT_DIR}/yolo_dataset',
    wall_seg_output_dir=f'{OUTPUT_DIR}/wall_segmentation',
    symbol_det_output_dir=f'{OUTPUT_DIR}/symbol_detection',
))

# ---- 通道 1: YOLO 格式 ----
print('=' * 60)
print('通道 1: 导出 YOLO 格式')
print('=' * 60)
yolo_result = pipeline.run_full(splits=SPLITS, target_format='yolo')

# ---- 通道 2a: 墙体分割数据 ----
print('=' * 60)
print('通道 2a: 导出墙体分割数据 (FPN 训练用)')
print('=' * 60)
wall_seg_result = pipeline.export_wall_segmentation_data(
    splits=SPLITS,
    crop_to_walls=True,  # 论文裁剪策略
)
print('墙体分割导出结果:', wall_seg_result)

# ---- 通道 2b: 门窗检测数据 ----
print('=' * 60)
print('通道 2b: 导出门窗检测数据 (Faster R-CNN 训练用)')
print('=' * 60)
symbol_det_result = pipeline.export_symbol_detection_data(
    splits=SPLITS,
    crop_to_walls=True,  # 论文裁剪策略
)
print('门窗检测导出结果:', symbol_det_result)

# ---- 验证输出 ----
print('\n' + '=' * 60)
print('导出完成！检查输出目录:')
print('=' * 60)

dirs_to_check = [
    f'{OUTPUT_DIR}/yolo_dataset',
    f'{OUTPUT_DIR}/yolo_dataset/train/images',
    f'{OUTPUT_DIR}/yolo_dataset/train/labels',
    f'{OUTPUT_DIR}/wall_segmentation/train/images',
    f'{OUTPUT_DIR}/wall_segmentation/train/masks',
    f'{OUTPUT_DIR}/symbol_detection/train/images',
    f'{OUTPUT_DIR}/symbol_detection/train',
]

for d in dirs_to_check:
    p = Path(d)
    if p.exists():
        if p.is_dir():
            count = len(list(p.glob('*')))
            print(f'  ✅ {d}  ({count} 个文件)')
        else:
            print(f'  ✅ {d}')
    else:
        print(f'  ❌ {d}  (不存在)')

# 检查关键文件
yaml_path = Path(f'{OUTPUT_DIR}/yolo_dataset/dataset.yaml')
if yaml_path.exists():
    print(f'\n  YOLO dataset.yaml 内容:')
    print(f'  {yaml_path.read_text()[:500]}')

ann_path = Path(f'{OUTPUT_DIR}/symbol_detection/train/annotations.json')
if ann_path.exists():
    import json
    with open(ann_path) as f:
        anns = json.load(f)
    total_doors = sum(l for a in anns for l in a['labels'] if l == 2)
    total_windows = sum(1 for a in anns for l in a['labels'] if l == 1)
    print(f'\n  门窗检测标注: {len(anns)} 张图, {total_doors} 个门, {total_windows} 个窗')


# ============================================================================
# 导出完成后的目录结构
# ============================================================================

"""
运行完成后，你的 output/ 目录会是这样的：

output/
├── coco_annotations/                   ← COCO 中间格式
│   ├── coco_annotations_train.json
│   └── coco_annotations_val.json
│
├── yolo_dataset/                       ← 通道 1: YOLO 训练数据
│   ├── dataset.yaml                    ← YOLO 训练配置文件
│   ├── train/
│   │   ├── images/                     ← 平面图图像
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   └── labels/                     ← YOLO 格式标签 (类别 + 归一化 polygon)
│   │       ├── 000001.txt
│   │       └── ...
│   └── val/
│       ├── images/
│       └── labels/
│
├── wall_segmentation/                  ← 通道 2a: 墙体分割训练数据
│   ├── filelist_train.txt
│   ├── filelist_val.txt
│   ├── train/
│   │   ├── images/                     ← 裁剪后的平面图
│   │   │   ├── 000000.png
│   │   │   └── ...
│   │   └── masks/                      ← 对应的墙体二值掩码 (0/255)
│   │       ├── 000000.png
│   │       └── ...
│   └── val/
│       ├── images/
│       └── masks/
│
└── symbol_detection/                   ← 通道 2b: 门窗检测训练数据
    ├── train/
    │   ├── images/                     ← 裁剪后的平面图
    │   │   ├── 000000.png
    │   │   └── ...
    │   └── annotations.json            ← bbox + label (1=window, 2=door)
    └── val/
        ├── images/
        └── annotations.json

接下来：
  - 把整个 output/ 目录上传到 Google Cloud Storage
  - 在 Cloud GPU 实例上运行 main.py --mode train_only
  - 或者在本地有 GPU 的话直接训练
"""
