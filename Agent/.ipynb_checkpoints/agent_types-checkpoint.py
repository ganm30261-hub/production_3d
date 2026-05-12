# agent_types.py
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

class AgentState(Enum):
    """Agent 当前所处的处理阶段"""
    IDLE            = "idle"
    THINKING        = "thinking"       # 正在做 CoT 规划
    ACTING          = "acting"         # 正在调用工具
    OBSERVING       = "observing"      # 正在验证结果
    REFLECTING      = "reflecting"     # IoU 不达标，进入反思
    EXPLORING       = "exploring"      # ToT 多分支并行搜索
    BACKTRACKING    = "backtracking"   # Go-Explore 回溯
    DONE            = "done"
    FAILED          = "failed"

@dataclass
class Thought:
    """CoT 推理步骤，每次 Act 前必须有 Thought"""
    reasoning:    str              # 为什么要做这件事
    plan:         List[str]        # 拆解后的子步骤
    tool_choice:  str              # 决定调用哪个工具
    confidence:   float            # 0-1，置信度

@dataclass
class Action:
    tool_name:    str
    tool_args:    Dict[str, Any]
    thought:      Thought          # 关联的推理

@dataclass
class Observation:
    success:      bool
    metrics:      Dict[str, float] # iou_mask, iou_vect, det_f1 等
    raw_output:   Any
    failure_reason: Optional[str]  # 失败时的诊断

@dataclass
class AgentStep:
    """ReAct 的一个完整步骤，写入 audit_log"""
    step_id:      int
    state:        AgentState
    thought:      Thought
    action:       Action
    observation:  Observation
    timestamp:    str

@dataclass
class AgentMemory:
    """Agent 运行时的工作记忆"""
    image_path:   str
    steps:        List[AgentStep]  = field(default_factory=list)
    current_mask: Any              = None  # 当前最优 mask
    best_iou:     float            = 0.0
    retry_count:  Dict[str, int]   = field(default_factory=dict)
    checkpoints:  Dict[str, Any]   = field(default_factory=dict)  # Go-Explore 回溯点
    tot_branches: List[Dict]       = field(default_factory=list)   # ToT 搜索分支