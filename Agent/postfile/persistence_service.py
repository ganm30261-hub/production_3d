"""
persistence_service.py — 持久化存储服务（人工审核后执行）

职责：
  接收审核结果（task_id + 合格/不合格）：
    合格   → 将 staging/{id}/ 搬运到 archive/
             同步到 GCS 永久路径
    不合格 → 记录拒绝原因（数据闭环素材）
             清理临时文件

为什么单独一个文件？
  审核是异步的，和生成完全解耦。
  audit_log 的写入需要原子性保证，单文件好测试。

用法：
  python persistence_service.py --task_id abc123 --approve
  python persistence_service.py --task_id abc123 --reject --reason wall_missing
  python persistence_service.py --list_pending   # 查看待审核任务列表
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from postprocess_config import STAGING_CFG, StagingConfig

logger = logging.getLogger(__name__)

AUDIT_LOG_PATH = os.path.join(STAGING_CFG.staging_dir, 'audit_log.jsonl')


# ══════════════════════════════════════════════════════════════
# 核心操作
# ══════════════════════════════════════════════════════════════

def approve(
    task_id:        str,
    upload_to_gcs:  bool = False,
    cfg:            StagingConfig = STAGING_CFG,
) -> dict:
    """
    审核通过：将 staging/{task_id}/ 搬到 archive/

    步骤：
      1. 验证 staging 文件存在
      2. move → archive/
      3. 可选：同步到 GCS 永久路径
      4. 写 audit_log
    """
    staging_task_dir = os.path.join(cfg.staging_dir, task_id)
    archive_task_dir = os.path.join(cfg.archive_dir,  task_id)

    if not os.path.exists(staging_task_dir):
        return {'task_id': task_id, 'status': 'error',
                'reason': f'staging 目录不存在: {staging_task_dir}'}

    os.makedirs(cfg.archive_dir, exist_ok=True)

    try:
        shutil.move(staging_task_dir, archive_task_dir)
        logger.info(f'[{task_id}] 已归档: {archive_task_dir}')
    except Exception as e:
        return {'task_id': task_id, 'status': 'error', 'reason': str(e)}

    gcs_url = None
    if upload_to_gcs:
        try:
            gcs_url = _sync_to_gcs_archive(archive_task_dir, task_id, cfg)
        except Exception as e:
            logger.warning(f'[{task_id}] GCS 同步失败: {e}')

    record = {
        'task_id':    task_id,
        'action':     'approve',
        'timestamp':  time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'archive_dir': archive_task_dir,
        'gcs_url':    gcs_url,
    }
    _append_audit_log(record)
    return {'task_id': task_id, 'status': 'approved', 'archive_dir': archive_task_dir}


def reject(
    task_id:   str,
    reason:    str  = 'other',
    note:      Optional[str] = None,
    keep_file: bool = False,
    cfg:       StagingConfig = STAGING_CFG,
) -> dict:
    """
    审核不通过：记录原因（数据闭环），清理临时文件

    reason 必须是 StagingConfig.REJECTION_REASONS 之一：
      wall_missing / wall_shape_wrong / door_offset /
      window_offset / scale_wrong / geometry_invalid / other

    keep_file=True 时保留文件以供人工复查，默认删除。
    """
    if reason not in StagingConfig.REJECTION_REASONS:
        logger.warning(f'未知拒绝原因 "{reason}"，改为 "other"')
        reason = 'other'

    staging_task_dir = os.path.join(cfg.staging_dir, task_id)

    record = {
        'task_id':   task_id,
        'action':    'reject',
        'reason':    reason,
        'note':      note,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    _append_audit_log(record)

    if not keep_file and os.path.exists(staging_task_dir):
        shutil.rmtree(staging_task_dir, ignore_errors=True)
        logger.info(f'[{task_id}] 已删除 staging 文件  reason={reason}')
    else:
        logger.info(f'[{task_id}] 标记为不合格（保留文件）  reason={reason}')

    return {'task_id': task_id, 'status': 'rejected', 'reason': reason}


# ══════════════════════════════════════════════════════════════
# 查询与统计
# ══════════════════════════════════════════════════════════════

def list_pending(cfg: StagingConfig = STAGING_CFG) -> list:
    """列出所有待审核的 task（staging 目录下有 meta.json 且未归档）"""
    pending = []
    staging_dir = cfg.staging_dir
    if not os.path.exists(staging_dir):
        return []
    for task_id in os.listdir(staging_dir):
        meta_path = os.path.join(staging_dir, task_id, 'meta.json')
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            pending.append(meta)
    pending.sort(key=lambda x: x.get('elapsed', 0))
    return pending


def rejection_summary(cfg: StagingConfig = STAGING_CFG) -> dict:
    """
    统计拒绝原因分布（数据闭环分析入口）
    返回 {reason: count} + 总拒绝数
    可用于判断哪类错误最多，从而优先优化对应算法
    """
    if not os.path.exists(AUDIT_LOG_PATH):
        return {}
    counts: dict = {}
    with open(AUDIT_LOG_PATH) as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                if rec.get('action') == 'reject':
                    r = rec.get('reason', 'other')
                    counts[r] = counts.get(r, 0) + 1
            except json.JSONDecodeError:
                continue
    return {'counts': counts, 'total_rejected': sum(counts.values())}


# ══════════════════════════════════════════════════════════════
# 内部工具
# ══════════════════════════════════════════════════════════════

def _append_audit_log(record: dict):
    """原子追加 audit log（JSONL 格式，一行一条记录）"""
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    with open(AUDIT_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    logger.debug(f'audit_log 追加: {record["action"]} {record["task_id"]}')


def _sync_to_gcs_archive(local_dir: str, task_id: str,
                          cfg: StagingConfig) -> str:
    """把 archive 目录同步到 GCS 永久路径"""
    from google.cloud import storage as gcs_storage
    client = gcs_storage.Client()
    bucket = client.bucket(cfg.gcs_bucket)
    for file_path in Path(local_dir).rglob('*'):
        if not file_path.is_file():
            continue
        rel   = file_path.relative_to(local_dir)
        blob  = bucket.blob(f'{cfg.gcs_archive}/{task_id}/{rel}')
        blob.upload_from_filename(str(file_path))
    url = f'gs://{cfg.gcs_bucket}/{cfg.gcs_archive}/{task_id}/'
    logger.info(f'GCS 永久归档完成: {url}')
    return url


# ══════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    p = argparse.ArgumentParser(description='Floor Plan Persistence Service')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--approve',       action='store_true')
    g.add_argument('--reject',        action='store_true')
    g.add_argument('--list_pending',  action='store_true')
    g.add_argument('--rejection_summary', action='store_true')

    p.add_argument('--task_id', default=None)
    p.add_argument('--reason',  default='other',
                   choices=StagingConfig.REJECTION_REASONS)
    p.add_argument('--note',    default=None, help='审核备注')
    p.add_argument('--gcs',     action='store_true', help='同步到 GCS')
    p.add_argument('--keep',    action='store_true', help='拒绝时保留文件')
    args = p.parse_args()

    if args.list_pending:
        tasks = list_pending()
        print(f'待审核任务: {len(tasks)} 个')
        for t in tasks:
            print(f'  {t["task_id"]}  stats={t.get("stats")}')

    elif args.rejection_summary:
        summary = rejection_summary()
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    elif args.approve:
        if not args.task_id:
            print('--approve 需要 --task_id')
        else:
            result = approve(args.task_id, upload_to_gcs=args.gcs)
            print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.reject:
        if not args.task_id:
            print('--reject 需要 --task_id')
        else:
            result = reject(args.task_id, args.reason, args.note, args.keep)
            print(json.dumps(result, indent=2, ensure_ascii=False))
