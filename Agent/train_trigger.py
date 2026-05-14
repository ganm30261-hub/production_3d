# train_trigger.py
"""
支柱3：训练触发与调度策略

两种触发机制：
    数据驱动   黄金数据集新增 >= N 张时触发
    性能衰退   最近 K 张图的 VLM 通过率低于阈值时触发

触发后的三步资源锁定：
    Step1  杀掉所有推理进程，释放显存
    Step2  分配显存预算，获取 GPU 独占锁
    Step3  训练结束后释放显存，唤醒重建 Agent
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from config import logger, PseudoLabelConfig, CFG


# ══════════════════════════════════════════════════════════════
# 触发配置
# ══════════════════════════════════════════════════════════════

@dataclass
class TriggerConfig:
    # 数据驱动触发
    golden_pool_path:       str   = '/workspace/production_3d/outputs/golden_pool'
    data_trigger_n:         int   = 50    # 新增 N 张触发（初期小批次快速迭代）
    data_trigger_increment: int   = 50    # 每次训练后，下次触发的增量

    # 性能衰退触发
    perf_window:            int   = 50    # 观察最近 K 张图
    perf_vlm_pass_thresh:   float = 0.70  # VLM 通过率低于此值触发
    perf_s_total_thresh:    float = 0.60  # 连续 20 张 S_total 低于此值触发
    perf_consecutive_low:   int   = 20

    # 早停
    early_stop_patience:    int   = 8     # 连续 N 个 epoch 不上升就停

    # 状态文件（记录上次触发时的样本数）
    trigger_state_path:     str   = '/workspace/production_3d/outputs/trigger_state.json'


DEFAULT_TRIGGER_CFG = TriggerConfig()


# ══════════════════════════════════════════════════════════════
# 触发状态管理
# ══════════════════════════════════════════════════════════════

class TriggerState:
    """持久化触发状态，重启后不会重复触发已完成的训练。"""

    def __init__(self, path: str = DEFAULT_TRIGGER_CFG.trigger_state_path):
        self.path = path
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return {
            'last_trigger_count': 0,
            'next_trigger_count': DEFAULT_TRIGGER_CFG.data_trigger_n,
            'total_trainings':    0,
            'last_train_time':    None,
            'perf_log':           [],   # 最近 K 张图的 S_total 记录
        }

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self._data, f, indent=2)

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value
        self.save()

    def append_perf(self, s_total: float, vlm_passed: bool) -> None:
        log = self._data.get('perf_log', [])
        log.append({
            'timestamp':  time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            's_total':    s_total,
            'vlm_passed': vlm_passed,
        })
        # 只保留最近 100 条
        self._data['perf_log'] = log[-100:]
        self.save()


# ══════════════════════════════════════════════════════════════
# 触发检查
# ══════════════════════════════════════════════════════════════

def check_data_trigger(
    state: TriggerState,
    cfg:   TriggerConfig = DEFAULT_TRIGGER_CFG,
) -> tuple:
    """
    数据驱动触发检查。
    返回 (should_trigger, reason)
    """
    current_count = _count_golden_pool(cfg.golden_pool_path)
    next_threshold = state.get('next_trigger_count', cfg.data_trigger_n)

    if current_count >= next_threshold:
        return True, f'黄金数据集新增到 {current_count} 张（阈值 {next_threshold}）'
    return False, f'黄金数据集 {current_count}/{next_threshold} 张，未达阈值'


def check_perf_trigger(
    state: TriggerState,
    cfg:   TriggerConfig = DEFAULT_TRIGGER_CFG,
) -> tuple:
    """
    性能衰退触发检查。
    返回 (should_trigger, reason)
    """
    perf_log = state.get('perf_log', [])
    if len(perf_log) < cfg.perf_window:
        return False, f'性能日志不足 {len(perf_log)}/{cfg.perf_window} 条'

    recent = perf_log[-cfg.perf_window:]

    # 1. VLM 通过率
    vlm_pass_rate = sum(1 for r in recent if r.get('vlm_passed')) / len(recent)
    if vlm_pass_rate < cfg.perf_vlm_pass_thresh:
        return True, f'VLM 通过率 {vlm_pass_rate:.1%} < {cfg.perf_vlm_pass_thresh:.0%}'

    # 2. 连续低分
    last_n = perf_log[-cfg.perf_consecutive_low:]
    if len(last_n) >= cfg.perf_consecutive_low:
        all_low = all(r.get('s_total', 1) < cfg.perf_s_total_thresh for r in last_n)
        if all_low:
            return True, f'连续 {cfg.perf_consecutive_low} 张 S_total < {cfg.perf_s_total_thresh}'

    return False, '性能正常，无需触发'


# ══════════════════════════════════════════════════════════════
# 早停回调
# ══════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    早停：连续 patience 个 epoch 指标不上升时停止。
    集成到 train_one_version 的 epoch 循环里。
    """

    def __init__(self, patience: int = 8, min_delta: float = 0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = -float('inf')
        self.counter   = 0

    def step(self, metric: float) -> bool:
        """
        传入当前 epoch 的指标值。
        返回 True = 应该停止训练。
        """
        if metric > self.best + self.min_delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f'[EarlyStopping] 指标未提升  '
                f'counter={self.counter}/{self.patience}  '
                f'best={self.best:.4f}'
            )

        if self.counter >= self.patience:
            logger.info(f'[EarlyStopping] 触发早停  best={self.best:.4f}')
            return True
        return False


# ══════════════════════════════════════════════════════════════
# 训练调度器
# ══════════════════════════════════════════════════════════════

class TrainScheduler:
    """
    训练触发 + 三步资源锁定 + 训练后唤醒。

    用法：
        scheduler = TrainScheduler()

        # 每处理完一张图后调用
        scheduler.record(s_total=0.72, vlm_passed=True)

        # 检查是否需要触发训练
        if scheduler.should_train():
            scheduler.run_training(cfg, on_complete=lambda: wake_reconstruct())
    """

    def __init__(
        self,
        cfg:     TriggerConfig   = DEFAULT_TRIGGER_CFG,
        train_cfg: PseudoLabelConfig = None,
    ):
        self.cfg       = cfg
        self.train_cfg = train_cfg or CFG
        self.state     = TriggerState(cfg.trigger_state_path)

    def record(self, s_total: float, vlm_passed: bool) -> None:
        """每张图处理完后调用，更新性能日志。"""
        self.state.append_perf(s_total, vlm_passed)

    def should_train(self) -> tuple:
        """
        检查是否应该触发训练。
        返回 (should, reason)
        """
        # 数据驱动
        data_ok, data_reason = check_data_trigger(self.state, self.cfg)
        if data_ok:
            return True, data_reason

        # 性能衰退
        perf_ok, perf_reason = check_perf_trigger(self.state, self.cfg)
        if perf_ok:
            return True, perf_reason

        return False, f'{data_reason} | {perf_reason}'

    def run_training(
        self,
        on_complete: Optional[Callable] = None,
    ) -> dict:
        """
        三步资源锁定 + 训练 + 唤醒。

        Step1: 卸载推理模型，清理显存
        Step2: 获取 GPU 独占锁，分配显存
        Step3: 训练，完成后释放锁，调用 on_complete
        """
        from resource_manager import GPU_LOCK, clear_gpu_memory
        from training.trainer import train_one_version
        from experiment_tracker import ExperimentTracker, get_data_version_id

        logger.info('[TrainScheduler] ▶ 训练触发，开始三步资源锁定')

        # Step1: 卸载推理模型
        try:
            from skeleon.vram_scheduler import SCHEDULER
            SCHEDULER.unload_all()
        except Exception:
            pass
        clear_gpu_memory()
        logger.info('[TrainScheduler] Step1 完成：推理模型已卸载')

        # Step2: 获取 GPU 独占锁
        data_version = get_data_version_id(self.cfg.golden_pool_path)
        tracker      = ExperimentTracker(self.train_cfg)

        logger.info(f'[TrainScheduler] Step2：获取 GPU 锁  data_version={data_version}')

        result = {}
        try:
            with GPU_LOCK.acquire(owner='train_scheduler', timeout=300):
                tracker.start_run(data_version_id=data_version)

                # 支持早停的训练
                import dataclasses
                train_cfg = dataclasses.replace(
                    self.train_cfg,
                    max_epochs = self.train_cfg.max_epochs,
                )
                result = train_one_version(
                    self.train_cfg.dataset_version,
                    base_cfg=train_cfg,
                )

                # 及格线判定
                gate_result = tracker.finish_run(
                    final_metrics = {'val_iou': result.get('best_val_iou', 0)},
                    ckpt_path     = result.get('best_ckpt'),
                )
                result['gate'] = gate_result

        finally:
            # Step3: 释放显存，更新触发状态
            clear_gpu_memory()
            self._update_trigger_state()
            logger.info('[TrainScheduler] Step3 完成：显存已释放')

            # 唤醒重建 Agent
            if on_complete:
                logger.info('[TrainScheduler] 唤醒重建 Agent')
                on_complete()

        return result

    def _update_trigger_state(self) -> None:
        """训练完成后更新触发状态，设置下次触发阈值。"""
        current_count  = _count_golden_pool(self.cfg.golden_pool_path)
        next_threshold = current_count + self.cfg.data_trigger_increment

        self.state.set('last_trigger_count', current_count)
        self.state.set('next_trigger_count', next_threshold)
        self.state.set('total_trainings',    self.state.get('total_trainings', 0) + 1)
        self.state.set('last_train_time',    time.strftime('%Y-%m-%dT%H:%M:%SZ'))

        logger.info(
            f'[TrainScheduler] 状态更新  '
            f'total_trainings={self.state.get("total_trainings")}  '
            f'next_threshold={next_threshold}'
        )


# ── 内部工具 ──

def _count_golden_pool(path: str) -> int:
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) if f.endswith('.json')])
