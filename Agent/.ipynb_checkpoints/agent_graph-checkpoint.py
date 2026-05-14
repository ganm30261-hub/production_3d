# agent_graph.py
"""
LangGraph 状态机：替换 FloorplanAgent.run() 里的手写循环。

状态节点（对应 AgentState）：
    think      → CoT 规划，生成 Thought
    act        → 工具调用，生成 Observation
    observe    → 评估结果，决定下一步
    reflect    → IoU 不达标，生成反思并调整参数
    backtrack  → Go-Explore 回溯到上一个有效检查点
    finalize   → 综合评分 + 写报告 + FailureRAG

状态转移（边）：
    think   → act
    act     → observe
    observe → reflect      (IoU < 阈值 且 重试次数未满)
    observe → backtrack    (工具执行失败)
    observe → think        (需要继续但还未完成)
    observe → finalize     (generate_svg 成功 且 IoU 达标)
    observe → finalize     (重试次数耗尽)
    reflect → think        (调整参数后重新规划)
    backtrack → think      (回溯成功后重新规划)
    finalize → END
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np

from agent_types import (
    AgentMemory, AgentState, AgentStep, Action, Observation, Thought,
)
from evaluation import EvalWeights, EvalResult, evaluate
from thought_logger import ThoughtLogger
from failure_rag import FailureRAG
from config import logger, PseudoLabelConfig

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    logger.warning('[!] langgraph 未安装: pip install langgraph')


# ══════════════════════════════════════════════════════════════
# Graph State（LangGraph 要求 TypedDict）
# ══════════════════════════════════════════════════════════════

class GraphState(TypedDict):
    """
    贯穿整个 Graph 的共享状态。
    每个节点接收这个 dict，返回要更新的字段（partial update）。
    """
    # ── 输入（run_graph 时注入）──
    image_path:    str
    image_rgb:     Any                    # np.ndarray
    gt_mask:       Any                    # np.ndarray | None
    cfg:           Any                    # PseudoLabelConfig
    agent:         Any                    # FloorplanAgent 实例

    # ── 运行时状态 ──
    memory:        Any                    # AgentMemory
    current_thought: Optional[dict]       # Thought.to_dict()
    current_obs:   Optional[dict]         # Observation 字段
    reflection:    str                    # reflect 节点输出的反思文本
    step_id:       int

    # ── 路由（必须声明在 TypedDict，LangGraph 才能 partial update）──
    _route:        str

    # ── 结果 ──
    wall_boxes:    List
    openings:      List[dict]
    eval_result:   Optional[dict]         # EvalResult.to_dict()
    thought_log:   str                    # 报告文件路径
    final_result:  Optional[dict]         # finalize 节点输出


# ══════════════════════════════════════════════════════════════
# 节点函数
# ══════════════════════════════════════════════════════════════

def node_think(state: GraphState) -> dict:
    """
    Think 节点：调用 Claude CoT 规划，生成下一步 Thought。
    """
    agent    = state["agent"]
    memory   = state["memory"]
    situation = agent._describe_situation(memory)

    thought  = agent.think(memory, situation)
    logger.info(f'[graph:think] step={state["step_id"]}  tool={thought.tool_choice}  conf={thought.confidence:.2f}')

    return {
        "current_thought": {
            "reasoning":   thought.reasoning,
            "plan":        thought.plan,
            "tool_choice": thought.tool_choice,
            "confidence":  thought.confidence,
        },
    }


def node_act(state: GraphState) -> dict:
    """
    Act 节点：执行工具调用，返回 Observation。
    同时把 Go-Explore 检查点写入 memory。
    """
    agent   = state["agent"]
    memory  = state["memory"]
    t_dict  = state["current_thought"]
    thought = Thought(**t_dict)

    obs = agent.act(thought, memory)

    # 更新最优 mask
    if obs.metrics.get("iou_mask", 0) > memory.best_iou:
        memory.best_iou     = obs.metrics["iou_mask"]
        memory.current_mask = (
            obs.raw_output.get("wall_mask") if obs.raw_output else None
        )

    # 记录 AgentStep 到 memory
    step = AgentStep(
        step_id     = state["step_id"],
        state       = AgentState.ACTING,
        thought     = thought,
        action      = Action(tool_name=thought.tool_choice, tool_args={}, thought=thought),
        observation = obs,
        timestamp   = time.strftime('%Y-%m-%dT%H:%M:%SZ'),
    )
    memory.steps.append(step)

    # ThoughtLogger 记录
    tl: ThoughtLogger = state["agent"]._tl
    tl.log_step_from_agent_step(step)

    logger.info(
        f'[graph:act]   step={state["step_id"]}  '
        f'tool={thought.tool_choice}  '
        f'success={obs.success}  '
        f'metrics={obs.metrics}'
    )

    # 提取 wall_boxes / openings（generate_svg 步骤后更新）
    updates: dict = {
        "current_obs": {
            "success":        obs.success,
            "metrics":        obs.metrics,
            "raw_output":     None,    # 不序列化原始 ndarray
            "failure_reason": obs.failure_reason,
        },
        "step_id": state["step_id"] + 1,
        "memory":  memory,
    }
    if obs.raw_output and isinstance(obs.raw_output, dict):
        if "wall_boxes" in obs.raw_output:
            updates["wall_boxes"] = obs.raw_output["wall_boxes"]
        if "openings" in obs.raw_output:
            updates["openings"]   = obs.raw_output["openings"]

    return updates


def node_observe(state: GraphState) -> dict:
    """
    Observe 节点：纯判断，不做任何外部调用，不写路由。
    路由判断全部在条件边函数 route_after_observe() 里完成，
    避免 LangGraph partial update 的字段同步问题。
    """
    return {}   # 不更新任何字段，路由由条件边函数直接计算


def node_reflect(state: GraphState) -> dict:
    """
    Reflect 节点：生成反思文本，更新 retry_count，
    把失败信息注入下一次 think() 的上下文。
    """
    agent   = state["agent"]
    memory  = state["memory"]
    obs_d   = state["current_obs"]

    # 重建 Observation 对象给 reflect()
    obs = Observation(
        success        = obs_d["success"],
        metrics        = obs_d["metrics"],
        raw_output     = None,
        failure_reason = obs_d["failure_reason"],
    )
    reflection = agent.reflect(obs, memory)

    tool = state["current_thought"]["tool_choice"]
    memory.retry_count[tool] = memory.retry_count.get(tool, 0) + 1

    # 把反思文本追加到最后一个 step 的 thought，方便 think() 参考
    if memory.steps:
        last_step = memory.steps[-1]
        last_step.state = AgentState.REFLECTING

    logger.info(f'[graph:reflect] {reflection}')
    return {"reflection": reflection, "memory": memory}


def node_backtrack(state: GraphState) -> dict:
    """
    Backtrack 节点：回溯到上一个有效检查点（Go-Explore）。
    """
    agent   = state["agent"]
    memory  = state["memory"]
    tool    = state["current_thought"]["tool_choice"]
    target  = f"before_{tool}"

    ok = agent.go_explore_backtrack(memory, target)
    logger.info(f'[graph:backtrack] target={target}  ok={ok}  restored_iou={memory.best_iou:.3f}')

    return {"memory": memory, "reflection": f"回溯到 {target}，IoU 恢复为 {memory.best_iou:.3f}"}


def node_finalize(state: GraphState) -> dict:
    """
    Finalize 节点：综合评分 + ThoughtLogger 写报告 + FailureRAG。
    """
    agent       = state["agent"]
    memory      = state["memory"]
    image_path  = state["image_path"]
    image_rgb   = state["image_rgb"]
    gt_mask     = state["gt_mask"]
    cfg         = state["cfg"]
    wall_boxes  = state.get("wall_boxes", [])
    openings    = state.get("openings", [])

    pred_mask   = memory.current_mask if memory.current_mask is not None \
                  else np.zeros((1, 1), dtype=np.uint8)
    _gt_mask    = gt_mask if gt_mask is not None else np.zeros_like(pred_mask)
    _image_rgb  = image_rgb if image_rgb is not None \
                  else np.zeros((1, 1, 3), dtype=np.uint8)

    eval_result: EvalResult = evaluate(
        pred_mask  = pred_mask,
        gt_mask    = _gt_mask,
        wall_boxes = wall_boxes,
        openings   = openings,
        image_rgb  = _image_rgb,
        cfg        = cfg,
        weights    = agent.eval_weights,
        vlm_client = agent.vlm_client,
    )

    tl: ThoughtLogger = agent._tl
    tl.log_eval(eval_result.to_dict())
    tl.finish()
    log_path = os.path.join(agent.log_dir, f"{Path(image_path).stem}_thought_log.json")

    if agent.failure_rag and not eval_result.passed:
        agent.failure_rag.add(
            image_name       = Path(image_path).stem,
            situation        = agent._describe_situation(memory),
            eval_result      = eval_result.to_dict(),
            thought_log_path = log_path,
        )

    last_obs = memory.steps[-1].observation if memory.steps else None
    final_result = {
        "image_path":  image_path,
        "best_iou":    memory.best_iou,
        "n_steps":     len(memory.steps),
        "success":     last_obs.success if last_obs else False,
        "svg_path":    (
            last_obs.raw_output.get("svg_path")
            if last_obs and isinstance(last_obs.raw_output, dict) else None
        ),
        "eval":        eval_result.to_dict(),
        "thought_log": log_path,
    }
    logger.info(
        f'[graph:finalize] S_total={eval_result.s_total:.3f}  '
        f'{"✓ passed" if eval_result.passed else "✗ failed"}'
    )
    return {"eval_result": eval_result.to_dict(), "final_result": final_result}


# ══════════════════════════════════════════════════════════════
# 条件边路由函数
# ══════════════════════════════════════════════════════════════

def route_after_observe(state: GraphState) -> str:
    """
    条件边路由函数：直接从 state 计算路由，不依赖 _route 字段。

    规则（优先级从高到低）：
        1. 工具执行失败              → backtrack
        2. IoU 低且重试未满          → reflect
        3. generate_svg 成功且达标   → finalize
        4. 步数或重试耗尽            → finalize
        5. 其他                      → think（继续）
    """
    obs    = state.get("current_obs")
    memory = state.get("memory")
    t_dict = state.get("current_thought")

    # 防御性检查：初始状态还没有 obs，直接 think
    if obs is None or t_dict is None or memory is None:
        return "think"

    iou       = obs["metrics"].get("iou_mask", 0)
    tool      = t_dict["tool_choice"]
    retry_cnt = memory.retry_count.get(tool, 0)
    MAX_RETRY = 3

    if not obs["success"]:
        route = "backtrack"
    elif iou < 0.75 and retry_cnt < MAX_RETRY:
        route = "reflect"
    elif tool == "generate_svg" and obs["success"] and memory.best_iou >= 0.75:
        route = "finalize"
    elif retry_cnt >= MAX_RETRY or state.get("step_id", 0) >= 10:
        route = "finalize"
    else:
        route = "think"

    logger.info(f'[graph:route] iou={iou:.3f}  retry={retry_cnt}  step={state.get("step_id",0)}  → {route}')
    return route


# ══════════════════════════════════════════════════════════════
# Graph 构建
# ══════════════════════════════════════════════════════════════

def build_floorplan_graph():
    """
    构建并编译 LangGraph StateGraph。
    节点函数从模块命名空间动态读取，monkeypatch 后立即生效。

    测试时在调用前替换模块级变量即可：
        import agent_graph as ag
        ag.node_think = mock_fn
        ag.build_floorplan_graph()

    节点与边：
        think → act → observe ──(reflect)──→ reflect → think
                             ──(backtrack)──→ backtrack → think
                             ──(think)──────→ think
                             ──(finalize)───→ finalize → END
    """
    import agent_graph as _self   # 读取当前模块最新的函数绑定
    if not HAS_LANGGRAPH:
        raise RuntimeError('pip install langgraph')

    g = StateGraph(GraphState)

    # ── 注册节点（从模块命名空间读取，支持 monkeypatch）──
    g.add_node("think",     _self.node_think)
    g.add_node("act",       _self.node_act)
    g.add_node("observe",   node_observe)
    g.add_node("reflect",   node_reflect)
    g.add_node("backtrack", node_backtrack)
    g.add_node("finalize",  node_finalize)

    # ── 固定边 ──
    g.add_edge("think",     "act")
    g.add_edge("act",       "observe")
    g.add_edge("reflect",   "think")
    g.add_edge("backtrack", "think")
    g.add_edge("finalize",  END)

    # ── 条件边（observe → 分支）──
    g.add_conditional_edges(
        "observe",
        route_after_observe,
        {
            "think":     "think",
            "reflect":   "reflect",
            "backtrack": "backtrack",
            "finalize":  "finalize",
        },
    )

    # ── 入口 ──
    g.set_entry_point("think")

    return g.compile()


# ══════════════════════════════════════════════════════════════
# 公开入口
# ══════════════════════════════════════════════════════════════

def run_graph(
    agent,
    image_path:  str,
    image_rgb:   Optional[np.ndarray]  = None,
    gt_mask:     Optional[np.ndarray]  = None,
    wall_boxes:  Optional[list]        = None,
    openings:    Optional[list]        = None,
) -> dict:
    """
    用 LangGraph graph 替代 FloorplanAgent.run() 手写循环。

    典型用法：
        from agent_graph import run_graph
        result = run_graph(agent, image_path, image_rgb=img_rgb)

    参数与 FloorplanAgent.run() 完全兼容，可无缝切换。
    """
    graph = build_floorplan_graph()

    # ThoughtLogger 挂到 agent 上，方便各节点共用
    agent._tl = ThoughtLogger(image_path, agent.log_dir)
    agent._tl.start()

    initial_state: GraphState = {
        "image_path":      image_path,
        "image_rgb":       image_rgb,
        "gt_mask":         gt_mask,
        "cfg":             agent.cfg,
        "agent":           agent,
        "memory":          AgentMemory(image_path=image_path),
        "current_thought": None,
        "current_obs":     None,
        "reflection":      "",
        "step_id":         0,
        "wall_boxes":      wall_boxes or [],
        "openings":        openings   or [],
        "eval_result":     None,
        "thought_log":     "",
        "final_result":    None,
        "_route":          "think",
    }

    final_state = graph.invoke(initial_state)
    return final_state.get("final_result", {})
