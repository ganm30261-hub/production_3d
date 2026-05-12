"""
utils.py — 图像处理工具箱
所有纯 CV / 数学逻辑：预处理、增强、mask 提取、bbox 提取
不依赖模型，可独立复用
"""

import io
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy import genfromtxt
from PIL import Image

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 图片读取
# ══════════════════════════════════════════════════════════════

def read_image(path: str) -> np.ndarray:
    """读取图片，支持本地路径和 gs:// 路径"""
    if path.startswith('gs://'):
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        with fs.open(path, 'rb') as f:
            buf = np.frombuffer(f.read(), np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.imread(path)


def load_sample(data_folder: str, folder: str, cubicasa_root: str) -> dict:
    """
    加载图片 + SVG 标注，返回 image 和完整 seg tensor。
    cubicasa_root 需要提前加入 sys.path 并 os.chdir。
    """
    import sys
    if cubicasa_root not in sys.path:
        sys.path.insert(0, cubicasa_root)
    from floortrans.loaders.house import House

    folder   = folder.strip('/')
    img_path = os.path.join(data_folder, folder, 'F1_scaled.png')
    svg_path = os.path.join(data_folder, folder, 'model.svg')

    img = read_image(img_path)
    if img is None:
        raise FileNotFoundError(f'图像不存在: {img_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if svg_path.startswith('gs://'):
        import gcsfs
        import tempfile
        fs = gcsfs.GCSFileSystem()
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
            with fs.open(svg_path, 'rb') as f:
                tmp.write(f.read())
            tmp_path = tmp.name
        house = House(tmp_path, h, w)
        os.unlink(tmp_path)
    else:
        house = House(svg_path, h, w)

    seg = house.get_segmentation_tensor()   # (C, H, W)

    if seg.shape[0] < 2:
        pad = np.zeros((2 - seg.shape[0], h, w), dtype=seg.dtype)
        seg = np.concatenate([seg, pad], axis=0)

    return {'image': img, 'seg': seg}


# ══════════════════════════════════════════════════════════════
# 论文预处理（Section 3）
# ══════════════════════════════════════════════════════════════

def crop_to_wall_bbox(
    image: np.ndarray,
    seg:   np.ndarray,
    wall_class_id: int = 2,
    padding: int = 10,
    min_size: int = 64,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    裁剪图片到 GT wall mask 的 bounding box（Section 3）。
    返回 (image_crop, seg_crop)，无墙区域时返回 (None, None)。
    """
    h, w      = image.shape[:2]
    wall_mask = (seg[0] == wall_class_id)

    if wall_mask.sum() == 0:
        return None, None

    rows = np.where(wall_mask.any(axis=1))[0]
    cols = np.where(wall_mask.any(axis=0))[0]

    y1 = max(rows[0]  - padding, 0)
    y2 = min(rows[-1] + padding + 1, h)
    x1 = max(cols[0]  - padding, 0)
    x2 = min(cols[-1] + padding + 1, w)

    if (y2 - y1) < min_size or (x2 - x1) < min_size:
        return None, None

    return image[y1:y2, x1:x2], seg[:, y1:y2, x1:x2]


# ══════════════════════════════════════════════════════════════
# Mask / BBox 提取
# ══════════════════════════════════════════════════════════════

def extract_wall_mask(seg: np.ndarray, wall_class_id: int = 2) -> np.ndarray:
    """从 seg[0] 提取墙体二值 mask：wall=1, background=0"""
    return (seg[0] == wall_class_id).astype(np.uint8)


def extract_door_window_boxes(
    seg: np.ndarray,
    door_class_id:   int = 2,
    window_class_id: int = 1,
    min_area:        int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 seg[1] 提取门窗 bounding boxes。
    返回 boxes (N,4) [x1,y1,x2,y2], labels (N,) 1=door 2=window
    """
    boxes, labels = [], []

    for class_id, label in [(door_class_id, 1), (window_class_id, 2)]:
        mask = (seg[1] == class_id).astype(np.uint8)
        if mask.sum() == 0:
            continue
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, x + w, y + h])
            labels.append(label)

    if boxes:
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)


# ══════════════════════════════════════════════════════════════
# 数据增强（Section 2.1）
# ══════════════════════════════════════════════════════════════

def augment_sample(
    image: np.ndarray,
    seg:   np.ndarray,
    scale_range:      Tuple[float, float] = (0.5, 2.0),
    rotation_degrees: List[int]           = None,
    use_flip:         bool                = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    论文 Section 2.1: random scaling + random rotation。
    同时作用于 image 和 seg（所有通道）。
    """
    if rotation_degrees is None:
        rotation_degrees = [0, 90, 180, 270]

    h, w = image.shape[:2]

    # 随机缩放
    scale = random.uniform(*scale_range)
    new_h = max(int(h * scale), 1)
    new_w = max(int(w * scale), 1)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    new_seg = np.zeros((seg.shape[0], new_h, new_w), dtype=seg.dtype)
    for c in range(seg.shape[0]):
        new_seg[c] = cv2.resize(
            seg[c].astype(np.float32), (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(seg.dtype)
    seg = new_seg

    # 随机旋转（90 度倍数）
    angle = random.choice(rotation_degrees)
    if angle != 0:
        k     = angle // 90
        image = np.rot90(image, k).copy()
        seg   = np.rot90(seg, k, axes=(1, 2)).copy()

    # 随机水平翻转
    if use_flip and random.random() > 0.5:
        image = np.fliplr(image).copy()
        seg   = np.flip(seg, axis=2).copy()

    return image, seg


# ══════════════════════════════════════════════════════════════
# 图片类型自适应预处理
# ══════════════════════════════════════════════════════════════

def detect_image_type(image: np.ndarray) -> str:
    """根据统计特征自动判断图片来源类型"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    ch_std    = np.std([image[:, :, c].std() for c in range(3)])
    is_bw     = ch_std < 5.0

    blur      = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_std = np.std(gray - blur)
    is_phone  = noise_std > 0.03

    contrast  = gray.std()
    is_scan   = contrast < 0.25

    if is_bw:    return 'bw'
    if is_scan:  return 'scan'
    if is_phone: return 'phone'
    return 'cad'


def preprocess_phone(image: np.ndarray) -> np.ndarray:
    """手机拍摄：双边滤波去噪 + CLAHE + 伽马矫正"""
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    gamma = 1.2
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(equalized, table)


def preprocess_scan(image: np.ndarray) -> np.ndarray:
    """旧建筑扫描：CLAHE + 伽马矫正 + 自适应阈值去背景"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    gamma = 1.5
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(np.uint8)
    brightened = cv2.LUT(enhanced, table)
    gray   = cv2.cvtColor(brightened, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    result = brightened.copy()
    result[binary == 255] = 255
    return result


def preprocess_bw(image: np.ndarray) -> np.ndarray:
    """黑白平面图：灰度化 + Otsu 阈值 + 形态学清理 + 转回 RGB"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)


def preprocess_stitch(images: list) -> np.ndarray:
    """多张图片：SIFT 特征匹配 + 单应变换拼接"""
    if len(images) == 1:
        return images[0]

    bgr_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]
    sift       = cv2.SIFT_create(nfeatures=2000)
    result_bgr = bgr_images[0]

    for i in range(1, len(bgr_images)):
        img_base   = result_bgr
        img_next   = bgr_images[i]
        gray_base  = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
        gray_next  = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
        kp1, des1  = sift.detectAndCompute(gray_base, None)
        kp2, des2  = sift.detectAndCompute(gray_next, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            continue

        index_params  = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann   = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good    = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good) < 4:
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _    = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        if H is None:
            continue

        h1, w1  = img_base.shape[:2]
        h2, w2  = img_next.shape[:2]
        warped  = cv2.warpPerspective(img_next, H, (w1 + w2, max(h1, h2)))
        result_bgr = warped.copy()
        result_bgr[:h1, :w1] = img_base

    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def adaptive_preprocess(image: np.ndarray, image_type: str = 'auto') -> np.ndarray:
    """
    自适应预处理统一入口。
    image_type: 'auto' | 'cad' | 'phone' | 'scan' | 'bw'
    """
    if image_type == 'auto':
        image_type = detect_image_type(image)

    if   image_type == 'phone': return preprocess_phone(image)
    elif image_type == 'scan':  return preprocess_scan(image)
    elif image_type == 'bw':    return preprocess_bw(image)
    else:                       return image   # 'cad' 不做额外处理


# ══════════════════════════════════════════════════════════════
# GCS 预处理缓存读取（GPU 训练时调用）
# ══════════════════════════════════════════════════════════════

def load_preprocessed_from_gcs(
    folder:      str,
    gcs_bucket:  str,
    gcs_prefix:  str,
    cache_dir:   str = '/tmp/floorplan_cache',
) -> Optional[np.ndarray]:
    """
    从 GCS 下载预处理后的图片，命中本地缓存则直接读取。
    返回 RGB ndarray；GCS 上不存在时返回 None。
    """
    import hashlib
    from google.cloud import storage as gcs_storage

    folder    = folder.strip('/')
    blob_name = f'{gcs_prefix}/{folder}/F1_preprocessed.png'
    cache_key = hashlib.md5(blob_name.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f'{cache_key}.png')

    if os.path.exists(cache_path):
        img = cv2.imread(cache_path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        client    = gcs_storage.Client()
        bucket    = client.bucket(gcs_bucket)
        blob      = bucket.blob(blob_name)
        if not blob.exists():
            return None
        img_bytes = blob.download_as_bytes()
        img_array = np.array(Image.open(io.BytesIO(img_bytes)))
        os.makedirs(cache_dir, exist_ok=True)
        cv2.imwrite(cache_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        return img_array
    except Exception as e:
        logger.warning(f'GCS 下载失败 {blob_name}: {e}')
        return None


# ══════════════════════════════════════════════════════════════
# CPU 端批量预处理 + 上传 GCS
# ══════════════════════════════════════════════════════════════

def _encode_image(image_rgb: np.ndarray) -> bytes:
    pil_img = Image.fromarray(image_rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG', optimize=False)
    return buf.getvalue()


def _gcs_exists(bucket, blob_name: str) -> bool:
    return bucket.blob(blob_name).exists()


def preprocess_and_upload_one(
    folder:      str,
    data_folder: str,
    bucket,
    gcs_prefix:  str,
    image_type:  str  = 'auto',
    overwrite:   bool = False,
) -> dict:
    """单个样本预处理 + 上传，返回结果 dict"""
    folder = folder.strip('/')
    gcs_img_path  = f'{gcs_prefix}/{folder}/F1_preprocessed.png'
    gcs_meta_path = f'{gcs_prefix}/{folder}/meta.json'

    if not overwrite and _gcs_exists(bucket, gcs_img_path):
        return {'folder': folder, 'status': 'skipped'}

    try:
        img_path = os.path.join(data_folder, folder, 'F1_scaled.png')
        img_bgr  = cv2.imread(img_path)
        if img_bgr is None:
            return {'folder': folder, 'status': 'error', 'reason': 'image not found'}
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        detected_type = detect_image_type(img_rgb) if image_type == 'auto' else image_type
        img_processed = adaptive_preprocess(img_rgb, detected_type)
        img_bytes     = _encode_image(img_processed)

        bucket.blob(gcs_img_path).upload_from_string(img_bytes, content_type='image/png')

        meta = {
            'folder':          folder,
            'image_type':      detected_type,
            'original_shape':  list(img_rgb.shape),
            'processed_shape': list(img_processed.shape),
            'gcs_img_path':    gcs_img_path,
            'timestamp':       time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        }
        bucket.blob(gcs_meta_path).upload_from_string(
            json.dumps(meta), content_type='application/json'
        )
        return {'folder': folder, 'status': 'done', 'image_type': detected_type, 'shape': list(img_processed.shape)}

    except Exception as e:
        return {'folder': folder, 'status': 'error', 'reason': str(e)}


def batch_preprocess_and_upload(
    data_folder: str,
    split_files: List[str],
    gcs_bucket:  str,
    gcs_prefix:  str,
    image_type:  str  = 'auto',
    overwrite:   bool = False,
    max_workers: int  = 8,
) -> dict:
    """
    批量预处理并上传到 GCS，多线程并行。
    在 CPU 环境离线运行一次，GPU 训练时直接读取结果。
    """
    from google.cloud import storage as gcs_storage
    client = gcs_storage.Client()
    bucket = client.bucket(gcs_bucket)

    all_folders = []
    for split_file in split_files:
        path = os.path.join(data_folder, split_file)
        if not os.path.exists(path):
            logger.warning(f'split 文件不存在: {path}')
            continue
        folders = genfromtxt(path, dtype='str').tolist()
        if isinstance(folders, str):
            folders = [folders]
        all_folders.extend(folders)

    logger.info(f'共 {len(all_folders)} 个样本，目标: gs://{gcs_bucket}/{gcs_prefix}/')

    stats, type_counter = {'done': 0, 'skipped': 0, 'error': 0}, {}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                preprocess_and_upload_one,
                folder, data_folder, bucket, gcs_prefix, image_type, overwrite
            ): folder
            for folder in all_folders
        }
        for future in as_completed(futures):
            result = future.result()
            status = result.get('status', 'error')
            stats[status] = stats.get(status, 0) + 1
            if status == 'done':
                t = result.get('image_type', 'unknown')
                type_counter[t] = type_counter.get(t, 0) + 1
            elif status == 'error':
                logger.error(f'ERROR {result["folder"]}: {result.get("reason")}')

    elapsed = time.time() - t0
    logger.info(f'完成！耗时 {elapsed:.1f}s  done={stats["done"]}  skip={stats["skipped"]}  err={stats["error"]}')

    summary = {
        'total':        len(all_folders),
        'stats':        stats,
        'type_counter': type_counter,
        'elapsed_sec':  round(elapsed, 1),
        'timestamp':    time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'image_type':   image_type,
        'gcs_prefix':   gcs_prefix,
    }
    client.bucket(gcs_bucket).blob(f'{gcs_prefix}/preprocess_summary.json').upload_from_string(
        json.dumps(summary, indent=2)
    )
    return summary
