# observability.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

可观测性：支柱5 —— Trace ID + 异常分类矩阵 + 结构化日志

每个任务有一个 trace_id（来自 PipelineState），
所有日志、异常、指标都带上这个 ID，方便跨 Agent 追踪。

异常分类矩阵：
    TypeError / AttributeError  → ScriptError    → 直接阻断，修代码
    ValidationError             → DataError      → 记录 Warning，跳过该图
    RuntimeError（显存）        → HardwareError  → 重启当前步骤
    其他 Exception              → UnknownError   → 记录，决策权交给 orchestrator
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import logging
import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

from config import logger


# ══════════════════════════════════════════════════════════════
# 异常分类
# ══════════════════════════════════════════════════════════════

class ErrorCategory(str, Enum):
    SCRIPT   = "script_error"    # TypeError/AttributeError → 修代码，直接阻断
    DATA     = "data_error"      # ValidationError → 跳过该图，不影响其他
    HARDWARE = "hardware_error"  # OOM/CUDA → 清显存后重试
    UNKNOWN  = "unknown_error"   # 其他，记录后交 orchestrator 决策


@dataclass
class ClassifiedError:
    category:       ErrorCategory
    original_error: Exception
    message:        str
    traceback_str:  str
    trace_id:       Optional[str] = None
    stage:          Optional[str] = None
    timestamp:      str = field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%SZ'))

    def to_dict(self) -> dict:
        return {
            'category':   self.category.value,
            'message':    self.message,
            'traceback':  self.traceback_str,
            'trace_id':   self.trace_id,
            'stage':      self.stage,
            'timestamp':  self.timestamp,
        }

    def should_abort(self) -> bool:
        """ScriptError 直接阻断整个进程，其他交 orchestrator 处理。"""
        return self.category == ErrorCategory.SCRIPT

    def should_retry(self) -> bool:
        return self.category == ErrorCategory.HARDWARE

    def should_skip(self) -> bool:
        return self.category == ErrorCategory.DATA


def classify_error(
    exc:      Exception,
    trace_id: str = None,
    stage:    str = None,
) -> ClassifiedError:
    """
    把任意异常分类到 ErrorCategory。

    分类规则：
        TypeError / AttributeError / ImportError  → SCRIPT
        ValidationError / ValueError / KeyError   → DATA
        RuntimeError（含 CUDA / OOM 关键词）       → HARDWARE
        其他                                       → UNKNOWN
    """
    from contracts import ValidationError as ContractValidationError

    tb_str = traceback.format_exc()
    msg    = str(exc)

    if isinstance(exc, (TypeError, AttributeError, ImportError)):
        cat = ErrorCategory.SCRIPT

    elif isinstance(exc, (ContractValidationError, ValueError, KeyError)):
        cat = ErrorCategory.DATA

    elif isinstance(exc, RuntimeError) and any(
        kw in msg.lower() for kw in ('cuda', 'oom', 'out of memory', 'device')
    ):
        cat = ErrorCategory.HARDWARE

    else:
        cat = ErrorCategory.UNKNOWN

    return ClassifiedError(
        category       = cat,
        original_error = exc,
        message        = msg,
        traceback_str  = tb_str,
        trace_id       = trace_id,
        stage          = stage,
    )


# ══════════════════════════════════════════════════════════════
# Trace Logger：带 trace_id 的结构化日志
# ══════════════════════════════════════════════════════════════

class TraceLogger:
    """
    给单个任务的日志加上 trace_id 前缀，并写入结构化事件文件。

    典型用法：
        tl = TraceLogger(trace_id="abc123", log_dir="/tmp/logs")
        tl.info("开始标注")
        tl.error("标注失败", exc=e)
        tl.metric("iou_pixel", 0.82)
        tl.finish()
    """

    def __init__(self, trace_id: str, log_dir: str = '/tmp/trace_logs'):
        self.trace_id  = trace_id
        self.log_dir   = log_dir
        self._events:  list = []
        self._t0:      float = time.time()
        os.makedirs(log_dir, exist_ok=True)

    # ── 日志方法 ──

    def info(self, msg: str, **kwargs) -> None:
        self._log('INFO', msg, **kwargs)
        logger.info(f'[{self.trace_id}] {msg}')

    def warning(self, msg: str, **kwargs) -> None:
        self._log('WARNING', msg, **kwargs)
        logger.warning(f'[{self.trace_id}] {msg}')

    def error(self, msg: str, exc: Exception = None, **kwargs) -> None:
        extra = {}
        if exc:
            classified = classify_error(exc, self.trace_id)
            extra = {
                'error_category': classified.category.value,
                'error_message':  classified.message,
            }
        self._log('ERROR', msg, **{**extra, **kwargs})
        logger.error(f'[{self.trace_id}] {msg}' + (f': {exc}' if exc else ''))

    def metric(self, name: str, value: float, stage: str = None) -> None:
        """记录一个数值指标，写入结构化事件。"""
        self._log('METRIC', f'{name}={value}', metric_name=name,
                  metric_value=value, stage=stage)

    def stage_start(self, stage: str) -> None:
        self._log('STAGE_START', f'阶段开始: {stage}', stage=stage)
        logger.info(f'[{self.trace_id}] ▶ {stage}')

    def stage_end(self, stage: str, result: str, elapsed_s: float = 0) -> None:
        symbol = '✓' if result == 'success' else '✗'
        self._log('STAGE_END', f'阶段结束: {stage}  {result}',
                  stage=stage, result=result, elapsed_s=elapsed_s)
        logger.info(f'[{self.trace_id}] {symbol} {stage}  {elapsed_s:.1f}s')

    def finish(self) -> str:
        """把所有事件写入 JSONL 文件，返回文件路径。"""
        path = os.path.join(self.log_dir, f'{self.trace_id}_trace.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for event in self._events:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        total = round(time.time() - self._t0, 2)
        logger.info(f'[{self.trace_id}] Trace 完成  elapsed={total}s  events={len(self._events)}  → {path}')
        return path

    # ── 内部 ──

    def _log(self, level: str, msg: str, **kwargs) -> None:
        self._events.append({
            'trace_id':  self.trace_id,
            'level':     level,
            'message':   msg,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'elapsed_s': round(time.time() - self._t0, 3),
            **kwargs,
        })


# ══════════════════════════════════════════════════════════════
# 上下文管理器：自动捕获 + 分类异常
# ══════════════════════════════════════════════════════════════

@contextmanager
def traced_stage(
    trace_logger: TraceLogger,
    stage:        str,
    reraise_script_errors: bool = True,
):
    """
    包裹一个阶段的执行，自动：
        - 记录阶段开始/结束
        - 捕获并分类异常
        - ScriptError 默认 reraise（阻断）
        - 其他异常记录后重新抛出，由 orchestrator 决策

    用法：
        with traced_stage(tl, "labeling") as ctx:
            result = run_labeling(...)
            ctx.set_result("success")
    """
    t0 = time.time()
    trace_logger.stage_start(stage)

    class _Ctx:
        result = 'unknown'
        def set_result(self, r: str): self.result = r

    ctx = _Ctx()
    try:
        yield ctx
        if ctx.result == 'unknown':
            ctx.result = 'success'
        trace_logger.stage_end(stage, ctx.result, round(time.time() - t0, 2))
    except Exception as exc:
        classified = classify_error(exc, trace_logger.trace_id, stage)
        trace_logger.error(
            f'阶段 {stage} 异常: {classified.category.value}',
            exc=exc
        )
        trace_logger.stage_end(stage, 'failed', round(time.time() - t0, 2))

        if reraise_script_errors and classified.should_abort():
            logger.critical(
                f'[{trace_logger.trace_id}] SCRIPT ERROR，阻断执行: {exc}'
            )
            raise   # ScriptError 直接阻断

        raise   # 其他异常重新抛出，让 orchestrator 根据类别决策
