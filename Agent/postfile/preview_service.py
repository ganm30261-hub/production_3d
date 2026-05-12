"""
preview_service.py — 预览生成服务（暂存区入口）

职责：
  1. 接收图片路径（或 GCS 路径）
  2. 调用 post_service.run_postprocess 生成 .glb
  3. 将结果写入 staging 目录（24 小时自动过期）
  4. 返回 { task_id, preview_url, stats } 给前端

部署场景：
  - 作为 Cloud Run 的 HTTP 端点（配合 FastAPI）
  - 或作为 Pub/Sub 消费者（异步处理队列）

用法：
  python preview_service.py --image floor.png
  # 返回 task_id 和预览 URL
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from postprocess_config import STAGING_CFG, VECT_CFG, INF_CFG
from post_service import run_postprocess

logger = logging.getLogger(__name__)


def generate_preview(
    image_path:   str,
    task_id:      Optional[str] = None,
    upload_to_gcs: bool = False,
) -> dict:
    """
    生成 3D 预览并存入 staging 区

    返回：
    {
        task_id:     str,
        status:      'pending',
        preview_url: str,         # 本地路径 or GCS 预签名 URL
        glb_path:    str,
        stats:       {...}
    }
    """
    task_id = task_id or str(uuid.uuid4())[:12]
    staging_dir = os.path.join(STAGING_CFG.staging_dir, task_id)
    os.makedirs(staging_dir, exist_ok=True)

    logger.info(f'[{task_id}] 开始生成预览...')
    t0 = time.time()

    try:
        result = run_postprocess(
            image_path = image_path,
            output_dir = staging_dir,
            vect_cfg   = VECT_CFG,
            inf_cfg    = INF_CFG,
            export_3d  = True,
            task_id    = task_id,
        )
    except Exception as e:
        logger.error(f'[{task_id}] 后处理失败: {e}')
        return {'task_id': task_id, 'status': 'error', 'reason': str(e)}

    glb_path    = result.get('glb_path')
    preview_url = glb_path or ''

    # 可选：上传到 GCS Staging Bucket
    if upload_to_gcs and glb_path and os.path.exists(glb_path):
        try:
            preview_url = _upload_to_gcs_staging(glb_path, task_id)
        except Exception as e:
            logger.warning(f'[{task_id}] GCS 上传失败，使用本地路径: {e}')

    elapsed = round(time.time() - t0, 2)

    payload = {
        'task_id':     task_id,
        'status':      'pending',
        'preview_url': preview_url,
        'glb_path':    glb_path,
        'stats':       result['stats'],
        'elapsed':     elapsed,
    }

    # 写入 staging 元数据
    meta_path = os.path.join(staging_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(payload, f, indent=2)

    logger.info(f'[{task_id}] 预览生成完成  elapsed={elapsed}s  url={preview_url}')
    return payload


def _upload_to_gcs_staging(local_path: str, task_id: str) -> str:
    """上传到 GCS Staging Bucket，返回预签名 URL"""
    from google.cloud import storage as gcs_storage
    import datetime

    client    = gcs_storage.Client()
    bucket    = client.bucket(STAGING_CFG.gcs_bucket)
    blob_name = f'{STAGING_CFG.gcs_staging}/{task_id}/{Path(local_path).name}'
    blob      = bucket.blob(blob_name)
    blob.upload_from_filename(local_path, content_type='model/gltf-binary')

    url = blob.generate_signed_url(
        expiration=datetime.timedelta(seconds=STAGING_CFG.preview_url_ttl),
        method='GET',
        version='v4',
    )
    logger.info(f'GCS 上传成功: gs://{STAGING_CFG.gcs_bucket}/{blob_name}')
    return url


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--image',     required=True)
    p.add_argument('--gcs',       action='store_true', help='上传到 GCS')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    result = generate_preview(args.image, upload_to_gcs=args.gcs)
    print(json.dumps(result, indent=2, ensure_ascii=False))
