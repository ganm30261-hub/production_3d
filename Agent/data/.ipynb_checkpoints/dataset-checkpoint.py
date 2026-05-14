# data/dataset.py
import os
import random
from typing import List

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from numpy import genfromtxt
from torch.utils.data import Dataset

from config import logger, PseudoLabelConfig


# ══════════════════════════════════════════════════════════════
# Split 文件生成（首次使用时调用一次）
# ══════════════════════════════════════════════════════════════

def generate_hq_split_files(
    data_folder: str,
    overwrite:   bool = False,
) -> List[str]:
    """
    扫描原始 train/val/test.txt，按子文件夹名生成：
        {split}_hq.txt       ← high_quality 样本
        {split}_hq_arch.txt  ← high_quality_architectural 样本

    combined 版本不单独生成文件，由 FloorplanDataset 同时加载
    以上两个文件实现。

    参数:
        data_folder : CubiCasa5k 数据根目录
        overwrite   : True 时强制重新生成，False 时跳过已存在文件

    用法（首次使用）:
        generate_hq_split_files(CFG.data_folder, overwrite=False)
    """
    SUBFOLDER_MAP = {
        'high_quality':               'hq',
        'high_quality_architectural': 'hq_arch',
    }
    generated = []

    for split in ('train', 'val', 'test'):
        src = os.path.join(data_folder, f'{split}.txt')
        if not os.path.exists(src):
            logger.warning(f'[skip] {src} 不存在')
            continue

        all_folders = genfromtxt(src, dtype='str').tolist()
        if isinstance(all_folders, str):
            all_folders = [all_folders]

        buckets = {tag: [] for tag in SUBFOLDER_MAP.values()}
        for folder in all_folders:
            subfolder = folder.strip('/').split('/')[0]
            tag = SUBFOLDER_MAP.get(subfolder)
            if tag:
                buckets[tag].append(folder)

        for tag, folders in buckets.items():
            out = os.path.join(data_folder, f'{split}_{tag}.txt')
            if os.path.exists(out) and not overwrite:
                logger.info(f'[skip]  {os.path.basename(out):35s} 已存在')
                continue
            with open(out, 'w') as f:
                f.write('\n'.join(folders) + ('\n' if folders else ''))
            generated.append(out)
            logger.info(f'[done]  {os.path.basename(out):35s} {len(folders)} 个样本')

    return generated


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

class FloorplanDataset(Dataset):
    """
    CubiCasa5k 数据集，支持三个 dataset_version：
        hq       → 加载 {split}_hq.txt
        hq_arch  → 加载 {split}_hq_arch.txt
        combined → 同时加载两个文件，合并去重后训练

    每个样本返回一个随机 tile（大小 = cfg.tile_size）：
        image  : (3, H, W)  float32 tensor，已归一化
        mask   : (H, W)     long tensor，0=background 1=wall
        boxes  : (N, 4)     float32 tensor，门窗 GT bbox [x1,y1,x2,y2]
        labels : (N,)       int64 tensor，1=door 2=window

    tile_size 必须是 14 的倍数（DINOv2 patch_size=14），
    默认 518=37×14，tile_overlap=56=4×14，stride=462=33×14。
    """

    def __init__(self, cfg: PseudoLabelConfig, split: str = 'train'):
        self.cfg      = cfg
        self.is_train = (split == 'train')

        file_list = cfg.train_files if split == 'train' else cfg.val_files

        # ── 多文件合并加载 ──
        all_folders: List[str] = []
        for filename in file_list:
            fpath = os.path.join(cfg.data_folder, filename)
            if not os.path.exists(fpath):
                logger.warning(
                    f'split 文件不存在，跳过: {filename}  '
                    f'（先运行 generate_hq_split_files()）'
                )
                continue
            rows = genfromtxt(fpath, dtype='str').tolist()
            if isinstance(rows, str):
                rows = [rows]
            all_folders.extend(rows)

        # 去重并保持顺序（combined 两个文件可能有极少量重叠）
        seen, unique = set(), []
        for f in all_folders:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        self.folders = np.array(unique)

        logger.info(
            f'[{split}] version={cfg.dataset_version}  '
            f'files={file_list}  →  {len(self.folders)} 个样本'
        )

        # ── 数据增强 ──
        self.aug = self._build_aug(cfg)

    def _build_aug(self, cfg: PseudoLabelConfig) -> A.Compose:
        """训练集用增强，验证集只做归一化。"""
        shared_kwargs = dict(additional_targets={'mask': 'mask'})

        if self.is_train:
            return A.Compose([
                A.RandomRotate90(p=0.75),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(
                    scale=(0.7, 1.5), rotate=(-10, 10),
                    mode=cv2.BORDER_REFLECT_101, p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.4),
                A.GaussNoise(std_range=(0, 0.02 * 255), p=0.3),
                A.ImageCompression(quality_range=(70, 95), p=0.2),
                A.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
                ToTensorV2(),
            ], **shared_kwargs)
        else:
            return A.Compose([
                A.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
                ToTensorV2(),
            ], **shared_kwargs)

    # ── Dataset protocol ──

    def __len__(self) -> int:
        return len(self.folders)

    def __getitem__(self, idx: int) -> dict:
        try:
            return self._load(self.folders[idx])
        except Exception as e:
            logger.warning(f'{self.folders[idx]}: {e}')
            return self._empty()

    # ── 内部加载逻辑 ──

    def _load(self, folder: str) -> dict:
        from floortrans.loaders.house import House

        folder   = folder.strip('/')
        img_path = os.path.join(self.cfg.data_folder, folder, 'F1_scaled.png')
        svg_path = os.path.join(self.cfg.data_folder, folder, 'model.svg')

        # ── 读图 ──
        img_bgr = cv2.imread(img_path)
        assert img_bgr is not None, f'图片不存在: {img_path}'
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_rgb.shape[:2]

        # ── 从 SVG 解析 wall mask 和门窗 bbox ──
        seg       = House(svg_path, h, w).get_segmentation_tensor()
        wall_mask = (seg[0] == self.cfg.wall_class_id).astype(np.uint8)

        boxes, labels = [], []
        for cls_id, lbl in [
            (self.cfg.door_class_id,   1),
            (self.cfg.window_class_id, 2),
        ]:
            m = (seg[1] == cls_id).astype(np.uint8)
            for cnt in cv2.findContours(
                m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]:
                if cv2.contourArea(cnt) < self.cfg.min_bbox_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                boxes.append([x, y, x + bw, y + bh])
                labels.append(lbl)

        # ── 随机采样 tile ──
        ts     = self.cfg.tile_size
        stride = ts - self.cfg.tile_overlap

        ys = list(range(0, max(h - ts + 1, 1), stride))
        xs = list(range(0, max(w - ts + 1, 1), stride))
        if not ys or ys[-1] + ts < h:
            ys.append(max(h - ts, 0))
        if not xs or xs[-1] + ts < w:
            xs.append(max(w - ts, 0))

        ty = random.choice(list(set(ys)))
        tx = random.choice(list(set(xs)))

        # ── 裁剪 + padding（边缘 tile 可能不足 ts×ts）──
        img_t  = img_rgb[ty:ty + ts, tx:tx + ts].copy()
        mask_t = wall_mask[ty:ty + ts, tx:tx + ts].copy()
        th, tw = img_t.shape[:2]

        if th < ts or tw < ts:
            img_t = cv2.copyMakeBorder(
                img_t, 0, ts - th, 0, ts - tw, cv2.BORDER_REFLECT_101
            )
            new_m = np.zeros((ts, ts), dtype=mask_t.dtype)
            new_m[:th, :tw] = mask_t
            mask_t = new_m

        img_t  = cv2.resize(img_t, (ts, ts))
        mask_t = cv2.resize(
            mask_t.astype(np.float32), (ts, ts),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)

        # ── 增强 ──
        aug     = self.aug(image=img_t, mask=mask_t)
        img_out = aug['image']
        msk_out = aug['mask'].long()

        # ── 把全图 bbox 变换到 tile 坐标系 ──
        tile_boxes, tile_labels = [], []
        for b, l in zip(boxes, labels):
            x1 = int(np.clip(b[0] - tx, 0, ts))
            y1 = int(np.clip(b[1] - ty, 0, ts))
            x2 = int(np.clip(b[2] - tx, 0, ts))
            y2 = int(np.clip(b[3] - ty, 0, ts))
            # 过滤掉 tile 外或过小的 bbox
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                tile_boxes.append([x1, y1, x2, y2])
                tile_labels.append(l)

        boxes_t = (
            torch.tensor(tile_boxes,  dtype=torch.float32)
            if tile_boxes else torch.zeros((0, 4))
        )
        labels_t = (
            torch.tensor(tile_labels, dtype=torch.int64)
            if tile_labels else torch.zeros(0, dtype=torch.int64)
        )

        return {
            'image':  img_out,
            'mask':   msk_out,
            'boxes':  boxes_t,
            'labels': labels_t,
        }

    def _empty(self) -> dict:
        """加载失败时返回全零样本，防止 DataLoader 崩溃。"""
        ts = self.cfg.tile_size
        return {
            'image':  torch.zeros(3, ts, ts),
            'mask':   torch.zeros(ts, ts).long(),
            'boxes':  torch.zeros((0, 4)),
            'labels': torch.zeros(0, dtype=torch.int64),
        }


# ══════════════════════════════════════════════════════════════
# Collate
# ══════════════════════════════════════════════════════════════

def collate_fn(batch: List[dict]) -> dict:
    """
    自定义 collate：image/mask stack 成 batch tensor，
    boxes/labels 保持 List[Tensor]（每张图的框数不同，不能 stack）。
    """
    return {
        'image':  torch.stack([b['image']  for b in batch]),
        'mask':   torch.stack([b['mask']   for b in batch]),
        'boxes':  [b['boxes']  for b in batch],
        'labels': [b['labels'] for b in batch],
    }
