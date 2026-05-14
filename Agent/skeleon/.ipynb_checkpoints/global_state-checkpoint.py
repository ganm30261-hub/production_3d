# global_state.py
"""
全局状态对象：支柱1 —— 状态机与接力机制

每个任务（一张图的完整处理流程）对应一个 PipelineState。
它是所有 Agent 之间的"交接包"，包含：
    - 任务身份（trace_id、image_path）
    - 当前阶段（stage）
    - 各 Agent 的输出路径
    - 成功/失败/重试 判定
    - 持久化到磁盘（断点续跑）

持久化策略：
    每次状态变更后立即写 {state_dir}/{trace_id}.json
    系统重启后调用 PipelineState.load(trace_id) 即可恢复
    已完成的阶段不会重跑（幂等）
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any

from config import logger


# ══════════════════════════════════════════════════════════════
# 阶段定义
# ══════════════════════════════════════════════════════════════

class Stage(str, Enum):
    """
    串行流水线的阶段顺序。
    str 混入使其可以直接序列化到 JSON。
    """
    INIT          = "init"           # 任务刚创建
    LABELING      = "labeling"       # 标注 Agent 运行中
    LABELED       = "labeled"        # 标注完成，等待训练
    TRAINING      = "training"       # 训练 Agent 运行中
    TRAINED       = "trained"        # 训练完成，等待重建
    RECONSTRUCTING = "reconstructing" # 3D 重建 Agent 运行中
    DONE          = "done"           # 全部完成
    FAILED        = "failed"         # 不可恢复的失败
    RETRYING      = "retrying"       # 某阶段失败，正在重试


# 合法的阶段转移表（只允许表中的转移）
VALID_TRANSITIONS: Dict[Stage, List[Stage]] = {
    Stage.INIT:           [Stage.LABELING,       Stage.FAILED],
    Stage.LABELING:       [Stage.LABELED,        Stage.RETRYING, Stage.FAILED],
    Stage.LABELED:        [Stage.TRAINING,       Stage.FAILED],
    Stage.TRAINING:       [Stage.TRAINED,        Stage.RETRYING, Stage.FAILED],
    Stage.TRAINED:        [Stage.RECONSTRUCTING, Stage.FAILED],
    Stage.RECONSTRUCTING: [Stage.DONE,           Stage.RETRYING, Stage.FAILED],
    Stage.RETRYING:       [Stage.LABELING, Stage.TRAINING, Stage.RECONSTRUCTING, Stage.FAILED],
    Stage.DONE:           [],
    Stage.FAILED:         [],
}


# ══════════════════════════════════════════════════════════════
# 判定标准
# ══════════════════════════════════════════════════════════════

class TransitionResult(str, Enum):
    SUCCESS = "success"
    FAIL    = "fail"
    RETRY   = "retry"


@dataclass
class StageResult:
    """
    每个阶段结束时的判定结果，决定状态转移方向。

    判定规则：
        SUCCESS → 进入下一阶段
        RETRY   → 重试当前阶段（retry_count < max_retries）
        FAIL    → 整个任务标记为 FAILED，写入 failure_reason
    """
    result:         TransitionResult
    metrics:        Dict[str, Any]        = field(default_factory=dict)
    failure_reason: Optional[str]         = None
    output_paths:   Dict[str, str]        = field(default_factory=dict)
    warnings:       List[str]             = field(default_factory=list)
    elapsed_s:      float                 = 0.0

    @staticmethod
    def success(metrics: dict = None, output_paths: dict = None,
                warnings: list = None, elapsed_s: float = 0.0) -> StageResult:
        return StageResult(
            result       = TransitionResult.SUCCESS,
            metrics      = metrics      or {},
            output_paths = output_paths or {},
            warnings     = warnings     or [],
            elapsed_s    = elapsed_s,
        )

    @staticmethod
    def retry(reason: str, metrics: dict = None) -> StageResult:
        return StageResult(
            result         = TransitionResult.RETRY,
            failure_reason = reason,
            metrics        = metrics or {},
        )

    @staticmethod
    def fail(reason: str, metrics: dict = None) -> StageResult:
        return StageResult(
            result         = TransitionResult.FAIL,
            failure_reason = reason,
            metrics        = metrics or {},
        )


# ══════════════════════════════════════════════════════════════
# 全局状态对象
# ══════════════════════════════════════════════════════════════

# 默认持久化目录
_STATE_DIR = os.environ.get(
    'PIPELINE_STATE_DIR',
    '/workspace/production_3d/outputs/pipeline_states'
)


@dataclass
class PipelineState:
    """
    一个任务（一张图）的完整生命周期状态。

    每次调用 transition() 后自动持久化到磁盘。
    系统重启后调用 PipelineState.load(trace_id) 恢复。

    字段：
        trace_id        唯一任务 ID，贯穿所有日志
        image_path      原始图片路径（GCS 或本地）
        stage           当前阶段
        retry_count     各阶段重试次数 {stage_name: count}
        max_retries     最大重试次数（默认3）
        stage_history   阶段变更历史
        outputs         各 Agent 的输出路径/结果
        warnings        所有 Warning 列表（不阻断流程）
        created_at      任务创建时间
        updated_at      最后更新时间
        metadata        扩展字段（调用方自由写入）
    """
    trace_id:      str
    image_path:    str
    stage:         Stage                   = Stage.INIT
    retry_count:   Dict[str, int]          = field(default_factory=dict)
    max_retries:   int                     = 3
    stage_history: List[Dict]              = field(default_factory=list)
    outputs:       Dict[str, Any]          = field(default_factory=dict)
    warnings:      List[str]               = field(default_factory=list)
    created_at:    str                     = field(default_factory=lambda: _now())
    updated_at:    str                     = field(default_factory=lambda: _now())
    metadata:      Dict[str, Any]          = field(default_factory=dict)

    # ── 状态转移 ──

    def transition(self, result: StageResult, next_stage: Stage) -> None:
        """
        执行状态转移，校验合法性，合并输出，持久化。

        参数：
            result     : 当前阶段的 StageResult
            next_stage : 目标阶段

        异常：
            ValueError  如果转移不合法（不在 VALID_TRANSITIONS 里）
        """
        allowed = VALID_TRANSITIONS.get(self.stage, [])
        if next_stage not in allowed:
            raise ValueError(
                f'非法状态转移: {self.stage} → {next_stage}  '
                f'允许: {[s.value for s in allowed]}'
            )

        # 记录历史
        self.stage_history.append({
            'from':          self.stage.value,
            'to':            next_stage.value,
            'result':        result.result.value,
            'metrics':       result.metrics,
            'failure_reason': result.failure_reason,
            'elapsed_s':     result.elapsed_s,
            'timestamp':     _now(),
        })

        # 合并输出路径
        self.outputs.update(result.output_paths)

        # 合并 warnings
        self.warnings.extend(result.warnings)

        # 更新重试计数
        if result.result == TransitionResult.RETRY:
            key = self.stage.value
            self.retry_count[key] = self.retry_count.get(key, 0) + 1

        self.stage      = next_stage
        self.updated_at = _now()

        logger.info(
            f'[{self.trace_id}] 状态转移: '
            f'{self.stage_history[-1]["from"]} → {next_stage.value}  '
            f'result={result.result.value}'
            + (f'  reason={result.failure_reason}' if result.failure_reason else '')
        )

        self.save()

    def can_retry(self, stage: Stage = None) -> bool:
        """当前阶段是否还可以重试。"""
        key = (stage or self.stage).value
        return self.retry_count.get(key, 0) < self.max_retries

    def is_stage_done(self, stage: Stage) -> bool:
        """
        某阶段是否已经成功完成（幂等判断）。
        断点续跑时用这个跳过已完成的阶段。
        """
        for h in self.stage_history:
            if h['from'] == stage.value and h['result'] == 'success':
                return True
        return False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(f'[{_now()}] {msg}')
        logger.warning(f'[{self.trace_id}] WARNING: {msg}')
        self.save()

    def set_output(self, key: str, value: Any) -> None:
        """写入某个 Agent 的输出，立即持久化。"""
        self.outputs[key] = value
        self.updated_at   = _now()
        self.save()

    # ── 持久化 ──

    def save(self, state_dir: str = _STATE_DIR) -> str:
        """序列化到 {state_dir}/{trace_id}.json，返回文件路径。"""
        os.makedirs(state_dir, exist_ok=True)
        path = os.path.join(state_dir, f'{self.trace_id}.json')
        data = {
            'trace_id':      self.trace_id,
            'image_path':    self.image_path,
            'stage':         self.stage.value,
            'retry_count':   self.retry_count,
            'max_retries':   self.max_retries,
            'stage_history': self.stage_history,
            'outputs':       self.outputs,
            'warnings':      self.warnings,
            'created_at':    self.created_at,
            'updated_at':    self.updated_at,
            'metadata':      self.metadata,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, trace_id: str, state_dir: str = _STATE_DIR) -> PipelineState:
        """从磁盘恢复状态，用于断点续跑。"""
        path = os.path.join(state_dir, f'{trace_id}.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f'状态文件不存在: {path}')
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        data['stage'] = Stage(data['stage'])
        state = cls(**data)
        logger.info(
            f'[{trace_id}] 恢复状态: stage={state.stage.value}  '
            f'history={len(state.stage_history)} 步'
        )
        return state

    @classmethod
    def create(cls, image_path: str, metadata: dict = None,
               state_dir: str = _STATE_DIR) -> PipelineState:
        """创建新任务，自动生成 trace_id 并持久化。"""
        trace_id = str(uuid.uuid4())[:12]
        state    = cls(
            trace_id   = trace_id,
            image_path = image_path,
            metadata   = metadata or {},
        )
        state.save(state_dir)
        logger.info(f'[{trace_id}] 新任务创建: {image_path}')
        return state

    @classmethod
    def list_all(cls, state_dir: str = _STATE_DIR) -> List[Dict]:
        """列出所有任务的摘要（不全量加载）。"""
        if not os.path.exists(state_dir):
            return []
        result = []
        for fname in sorted(os.listdir(state_dir)):
            if not fname.endswith('.json'):
                continue
            try:
                with open(os.path.join(state_dir, fname)) as f:
                    d = json.load(f)
                result.append({
                    'trace_id':   d['trace_id'],
                    'image_path': d['image_path'],
                    'stage':      d['stage'],
                    'updated_at': d['updated_at'],
                    'n_warnings': len(d.get('warnings', [])),
                })
            except Exception:
                pass
        return result


# ── 内部工具 ──

def _now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ')
