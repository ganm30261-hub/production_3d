# experiment_tracker.py
"""
支柱2：MLflow 实验追踪账本

在现有 trainer.py 的基础上扩展：
    1. 锁死参数记录规范（LoRA rank/alpha + Data Version ID）
    2. 存储 Artifacts（权重 + 对比可视化图）
    3. 及格线判定（val_iou < 0.75 → FAILED）
    4. GCS 路径规范
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from config import logger, PseudoLabelConfig

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.warning('[!] mlflow 未安装: pip install mlflow')


# ══════════════════════════════════════════════════════════════
# 及格线配置
# ══════════════════════════════════════════════════════════════

@dataclass
class PassCriteria:
    """
    模型晋升及格线。
    低于任意一条线 → 标记 FAILED，不晋升。
    """
    min_val_iou:      float = 0.75   # 分割 IoU
    min_det_f1:       float = 0.60   # 门窗检测 F1（可选）
    max_train_loss:   float = 0.50   # 训练 loss 上限（防止过拟合异常）

    def check(self, metrics: dict) -> tuple:
        """
        返回 (passed, reason)
        """
        val_iou    = metrics.get('val_iou', 0)
        train_loss = metrics.get('train_loss', 999)

        if val_iou < self.min_val_iou:
            return False, f'val_iou={val_iou:.4f} < {self.min_val_iou}'
        if train_loss > self.max_train_loss:
            return False, f'train_loss={train_loss:.4f} > {self.max_train_loss}'
        return True, 'passed'


DEFAULT_PASS = PassCriteria()


# ══════════════════════════════════════════════════════════════
# Data Version ID
# ══════════════════════════════════════════════════════════════

def get_data_version_id(golden_pool_path: str) -> str:
    """
    生成本次训练的数据版本 ID。
    格式：{样本数}samples_{最新文件时间戳}
    确保每次训练都能追溯用了哪批数据。
    """
    if not os.path.exists(golden_pool_path):
        return 'v0_0samples'

    files = sorted([
        f for f in os.listdir(golden_pool_path) if f.endswith('.json')
    ])
    n = len(files)
    if not files:
        return 'v0_0samples'

    latest_ts = int(os.path.getmtime(
        os.path.join(golden_pool_path, files[-1])
    ))
    return f'v{time.strftime("%Y%m%d", time.localtime(latest_ts))}_{n}samples'


# ══════════════════════════════════════════════════════════════
# 对比可视化（模型预测 vs 伪标注）
# ══════════════════════════════════════════════════════════════

def make_comparison_image(
    image_rgb:    np.ndarray,
    pred_mask:    np.ndarray,
    label_mask:   np.ndarray,
    save_path:    str,
) -> str:
    """
    生成"模型预测 vs 伪标注"对比图，存为 PNG。
    左：预测 mask（红色叠加）
    右：伪标注 mask（绿色叠加）
    """
    import cv2

    H, W = image_rgb.shape[:2]
    left  = image_rgb.copy()
    right = image_rgb.copy()

    # 预测：红色
    overlay = left.copy()
    overlay[pred_mask == 1]  = [200, 50, 50]
    left  = cv2.addWeighted(left,  0.6, overlay, 0.4, 0)

    # 伪标注：绿色
    overlay = right.copy()
    overlay[label_mask == 1] = [50, 200, 50]
    right = cv2.addWeighted(right, 0.6, overlay, 0.4, 0)

    # 拼接 + 标题
    combined = np.hstack([left, right])
    cv2.putText(combined, 'Prediction',   (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,50,50),  2)
    cv2.putText(combined, 'Pseudo Label', (W+10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,200,50),  2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    return save_path


# ══════════════════════════════════════════════════════════════
# ExperimentTracker
# ══════════════════════════════════════════════════════════════

class ExperimentTracker:
    """
    MLflow 实验追踪，扩展 trainer.py 的基础日志。

    新增内容：
        - Data Version ID
        - 对比可视化 Artifact
        - 及格线判定
        - GCS Artifact 路径规范
    """

    def __init__(
        self,
        cfg:           PseudoLabelConfig,
        pass_criteria: PassCriteria      = DEFAULT_PASS,
        mlflow_dir:    str               = '/workspace/production_3d/mlruns',
    ):
        self.cfg           = cfg
        self.pass_criteria = pass_criteria
        self.mlflow_dir    = mlflow_dir
        self._run_id:      Optional[str] = None

    def start_run(
        self,
        data_version_id: str,
        extra_params:    dict = None,
    ) -> str:
        """
        开启一个 MLflow run，记录所有超参数。
        返回 run_id。
        """
        if not HAS_MLFLOW:
            logger.warning('[ExperimentTracker] MLflow 未安装，跳过追踪')
            return 'no_mlflow'

        mlflow.set_tracking_uri(f'file://{self.mlflow_dir}')
        mlflow.set_experiment(self.cfg.mlflow_experiment)

        run = mlflow.start_run(run_name=self.cfg.mlflow_run_name)
        self._run_id = run.info.run_id

        # 锁死参数记录规范
        params = {
            # 模型超参
            'lora_r':           self.cfg.lora_r,
            'lora_alpha':       self.cfg.lora_alpha,
            'lora_dropout':     self.cfg.lora_dropout,
            'lora_target':      str(self.cfg.lora_target),
            'learning_rate':    self.cfg.learning_rate,
            'batch_size':       self.cfg.batch_size,
            'max_epochs':       self.cfg.max_epochs,
            'warmup_epochs':    self.cfg.warmup_epochs,
            'weight_decay':     self.cfg.weight_decay,
            'backbone':         self.cfg.dinov2_model,
            # 数据版本
            'data_version_id':  data_version_id,
            'dataset_version':  self.cfg.dataset_version,
            'train_files':      str(self.cfg.train_files),
            # 训练策略
            'use_amp':          self.cfg.use_amp,
            'grad_clip':        self.cfg.grad_clip,
            'dice_weight':      self.cfg.dice_weight,
            'bce_weight':       self.cfg.bce_weight,
            'wall_class_weight': self.cfg.wall_class_weight,
            # 及格线
            'pass_min_val_iou': self.pass_criteria.min_val_iou,
            'pass_min_det_f1':  self.pass_criteria.min_det_f1,
        }
        if extra_params:
            params.update(extra_params)

        mlflow.log_params(params)
        logger.info(f'[ExperimentTracker] MLflow run 开启  run_id={self._run_id}')
        return self._run_id

    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """记录每个 epoch 的指标。"""
        if not HAS_MLFLOW or not self._run_id:
            return
        mlflow.log_metrics(metrics, step=epoch)

    def log_comparison(
        self,
        image_rgb:  np.ndarray,
        pred_mask:  np.ndarray,
        label_mask: np.ndarray,
        epoch:      int,
    ) -> None:
        """生成对比可视化图并上传到 MLflow。"""
        if not HAS_MLFLOW or not self._run_id:
            return
        save_path = f'/tmp/comparison_ep{epoch:03d}.png'
        make_comparison_image(image_rgb, pred_mask, label_mask, save_path)
        mlflow.log_artifact(save_path, artifact_path='comparisons')
        logger.info(f'[ExperimentTracker] 对比图已上传: ep{epoch}')

    def finish_run(
        self,
        final_metrics: dict,
        ckpt_path:     Optional[str] = None,
    ) -> dict:
        """
        结束 run，进行及格线判定，返回判定结果。
        """
        if not HAS_MLFLOW or not self._run_id:
            return {'passed': True, 'reason': 'no_mlflow'}

        passed, reason = self.pass_criteria.check(final_metrics)

        mlflow.log_params({
            'final_result': 'PASSED' if passed else 'FAILED',
            'fail_reason':  reason if not passed else '',
        })
        mlflow.log_metrics({'final_val_iou': final_metrics.get('val_iou', 0)})

        # 上传权重文件
        if ckpt_path and os.path.exists(ckpt_path):
            mlflow.log_artifact(ckpt_path, artifact_path='model')
            logger.info(f'[ExperimentTracker] 权重已上传: {os.path.basename(ckpt_path)}')

        mlflow.end_run()
        self._run_id = None

        result = {
            'passed':   passed,
            'reason':   reason,
            'run_id':   self._run_id,
            'gcs_path': self._gcs_artifact_path(),
        }
        symbol = '✓' if passed else '✗'
        logger.info(f'[ExperimentTracker] {symbol} 训练{"达标" if passed else "未达标"}  {reason}')
        return result

    def _gcs_artifact_path(self) -> str:
        """GCS Artifact 存储路径规范。"""
        lr_str  = f'{self.cfg.learning_rate:.0e}'.replace('-0', '-')
        return (
            f'gs://{self.cfg.gcs_bucket}/models/dinov2_lora/'
            f'{self.cfg.dataset_version}/'
            f'lora_r{self.cfg.lora_r}_lr{lr_str}/'
        )
