"""
preprocess_runner.py — CPU 端批量预处理入口

职责：读取原始 CubiCasa5k 数据 → 自适应预处理 → 上传 GCS
在 CPU 环境（本地 / 廉价 CPU 实例）运行一次即可，GPU 训练时直接读取结果。

用法：
  python preprocess_runner.py                        # 处理 train + val（默认跳过已存在）
  python preprocess_runner.py --overwrite            # 强制重新处理全部
  python preprocess_runner.py --splits val.txt       # 只处理 val
  python preprocess_runner.py --image_type scan      # 强制指定图片类型
  python preprocess_runner.py --validate             # 验证 GCS 上的处理结果
"""

import argparse
import logging
import os
import sys

from config import PaperConfig, CUBICASA_ROOT
from utils import batch_preprocess_and_upload

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='CubiCasa5k CPU 批量预处理')
    p.add_argument('--splits',     nargs='+', default=['train.txt', 'val.txt'],
                   help='要处理的 split 文件列表')
    p.add_argument('--image_type', default='auto',
                   choices=['auto', 'cad', 'phone', 'scan', 'bw'],
                   help='图片类型（auto=自动检测）')
    p.add_argument('--overwrite',  action='store_true',
                   help='强制重新处理已存在的文件')
    p.add_argument('--workers',    type=int, default=8,
                   help='并行线程数（建议=CPU 核心数）')
    p.add_argument('--validate',   action='store_true',
                   help='验证 GCS 上的预处理结果（不运行预处理）')
    p.add_argument('--n_check',    type=int, default=20,
                   help='验证时随机抽查的样本数')
    return p.parse_args()


def run_validation(cfg: PaperConfig, split_file: str, n_check: int):
    """验证 GCS 上的预处理结果（文件完整性 + 图片质量）"""
    import io
    import json
    import random as rnd

    import numpy as np
    from google.cloud import storage as gcs_storage
    from numpy import genfromtxt
    from PIL import Image

    client = gcs_storage.Client()
    bucket = client.bucket(cfg.gcs_bucket)

    folders = genfromtxt(os.path.join(cfg.data_folder, split_file), dtype='str').tolist()
    if isinstance(folders, str):
        folders = [folders]

    total       = len(folders)
    exist_count = sum(
        1 for f in folders
        if bucket.blob(f'{cfg.gcs_preprocessed}/{f.strip("/")}/F1_preprocessed.png').exists()
    )
    upload_rate = exist_count / total
    logger.info(f'上传率: {exist_count}/{total} = {upload_rate*100:.1f}%')

    check_folders = rnd.sample(folders, min(n_check, len(folders)))
    shape_ok = brightness_ok = contrast_ok = 0
    type_dist = {}

    for folder in check_folders:
        folder    = folder.strip('/')
        blob_name = f'{cfg.gcs_preprocessed}/{folder}/F1_preprocessed.png'
        meta_name = f'{cfg.gcs_preprocessed}/{folder}/meta.json'
        try:
            img_bytes = bucket.blob(blob_name).download_as_bytes()
            img       = np.array(Image.open(io.BytesIO(img_bytes)))
            meta      = json.loads(bucket.blob(meta_name).download_as_text())
            t         = meta.get('image_type', 'unknown')
            type_dist[t] = type_dist.get(t, 0) + 1

            if img.ndim == 3 and img.shape[2] == 3: shape_ok += 1
            if 5 < img.mean() < 250:                brightness_ok += 1
            if img.std() > 10:                      contrast_ok += 1
        except Exception as e:
            logger.warning(f'验证失败 {folder}: {e}')

    n = len(check_folders)
    logger.info(f'尺寸/通道正确: {shape_ok}/{n}')
    logger.info(f'亮度范围合理: {brightness_ok}/{n}')
    logger.info(f'对比度足够:   {contrast_ok}/{n}')
    logger.info(f'图片类型分布: {type_dist}')


def main():
    args = parse_args()

    os.chdir(CUBICASA_ROOT)
    sys.path.insert(0, CUBICASA_ROOT)

    cfg = PaperConfig()

    if args.validate:
        for split in args.splits:
            logger.info(f'=== 验证 {split} ===')
            run_validation(cfg, split, args.n_check)
        return

    logger.info(f'开始批量预处理  splits={args.splits}  type={args.image_type}  '
                f'workers={args.workers}  overwrite={args.overwrite}')

    summary = batch_preprocess_and_upload(
        data_folder  = cfg.data_folder,
        split_files  = args.splits,
        gcs_bucket   = cfg.gcs_bucket,
        gcs_prefix   = cfg.gcs_preprocessed,
        image_type   = args.image_type,
        overwrite    = args.overwrite,
        max_workers  = args.workers,
    )

    logger.info(f'预处理完成: {summary["stats"]}')


if __name__ == '__main__':
    main()
