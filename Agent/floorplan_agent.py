# floorplan_agent.py
import json
import time
from typing import Optional

import anthropic
import numpy as np

from agent_types import (
    AgentMemory, AgentState, AgentStep, Action, Observation, Thought,
)
from agent_tools import TOOL_REGISTRY, ToolRAG
from evaluation import evaluate, EvalWeights, EvalResult
from thought_logger import ThoughtLogger
from failure_rag import FailureRAG

REACT_SYSTEM_PROMPT = """
你是一位专业的建筑图纸分析 Agent。每次行动必须严格按照 JSON 格式输出：
{
  "thought": {
    "reasoning": "为什么做这一步",
    "plan": ["子步骤1", "子步骤2"],
    "tool_choice": "工具名称",
    "confidence": 0.85
  },
  "action": {
    "tool_name": "工具名称",
    "tool_args": {}
  }
}

处理流程：
1. 识别外墙骨架（高置信度墙体）
2. 分割内部房间边界
3. 插入门窗细部

失败反思规则：
- IoU < 0.75：记录"墙体边界模糊，降低 iou_threshold 重试"
- 检测框数量 = 0：记录"门窗未检测到，降低 det_score_thresh 重试"
- 3D 布尔失败：记录"几何冲突，回溯到矢量化阶段重试"
最多重试 3 次，第 4 次直接标记为 FAILED。
"""


class FloorplanAgent:

    def __init__(
        self,
        cfg,
        model:          str               = "claude-sonnet-4-20250514",
        log_dir:        str               = "./thought_logs",
        failure_rag:    Optional[FailureRAG] = None,
        eval_weights:   Optional[EvalWeights] = None,
        vlm_client      = None,           # 语义评分用，可与 self.client 共用
    ):
        self.cfg          = cfg
        self.client       = anthropic.Anthropic()
        self.model        = model
        self.tool_rag     = ToolRAG()
        self.log_dir      = log_dir
        self.failure_rag  = failure_rag
        self.eval_weights = eval_weights or EvalWeights()
        self.vlm_client   = vlm_client or self.client

    # ──────────────────────────────────────────
    # ReAct 三步
    # ──────────────────────────────────────────

    def think(self, memory: AgentMemory, situation: str) -> Thought:
        """让 Claude 做 CoT 规划，返回结构化 Thought。"""
        relevant_tools = self.tool_rag.retrieve(situation, top_k=3)
        tools_desc = "\n".join(
            f"- {name}: {TOOL_REGISTRY[name].description}"
            for name in relevant_tools
        )
        # ── 接入 FailureRAG：真实 few-shot 替代 stub ──
        few_shot = (
            self.failure_rag.retrieve_as_fewshot(situation, top_k=3)
            if self.failure_rag and len(self.failure_rag) > 0
            else "（错题集为空，暂无参考案例）"
        )

        prompt = f"""
当前情况：{situation}
历史步骤数：{len(memory.steps)}
当前最优 IoU：{memory.best_iou:.3f}
当前重试次数：{memory.retry_count}

可用工具（RAG 检索）：
{tools_desc}

参考案例（错题集）：
{few_shot}

请输出下一步的 JSON 决策。
"""
        response = self.client.messages.create(
            model      = self.model,
            max_tokens = 512,
            system     = REACT_SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": prompt}],
        )
        data = json.loads(response.content[0].text)
        return Thought(**data["thought"])

    def act(self, thought: Thought, memory: AgentMemory) -> Observation:
        """执行工具调用，返回 Observation。"""
        tool = TOOL_REGISTRY.get(thought.tool_choice)
        if tool is None:
            return Observation(
                success=False, metrics={}, raw_output=None,
                failure_reason=f"工具 {thought.tool_choice} 不存在",
            )
        if tool.function is None:
            return Observation(
                success=False, metrics={}, raw_output=None,
                failure_reason=f"工具 {thought.tool_choice} 未绑定（检查 vector_logic import）",
            )
        try:
            memory.checkpoints[f"before_{thought.tool_choice}"] = {
                "mask":     memory.current_mask.copy() if memory.current_mask is not None else None,
                "best_iou": memory.best_iou,
            }
            tool_args = thought.tool_args if hasattr(thought, "tool_args") else {}
            result    = tool.function(**tool_args)
            metrics   = self._evaluate(result, memory)
            return Observation(
                success        = metrics.get("iou_mask", 1.0) > 0.0,
                metrics        = metrics,
                raw_output     = result,
                failure_reason = None,
            )
        except Exception as e:
            return Observation(
                success=False, metrics={}, raw_output=None,
                failure_reason=str(e),
            )

    def reflect(self, obs: Observation, memory: AgentMemory) -> str:
        """IoU 不达标时生成反思文本，决定下一步策略。"""
        iou = obs.metrics.get("iou_mask", 0)
        if iou < 0.75:
            return (
                f"检测到墙体拓扑断裂（IoU={iou:.3f}），"
                f"建议降低 iou_threshold 并重新运行 shrinking_algorithm"
            )
        if obs.metrics.get("det_f1", 1.0) < 0.5:
            return "门窗检测 F1 过低，建议降低 det_score_thresh 重试"
        if not obs.success:
            return f"工具执行失败：{obs.failure_reason}，尝试回溯到上一个有效状态"
        return ""

    # ──────────────────────────────────────────
    # 高级搜索策略
    # ──────────────────────────────────────────

    def tot_search(self, memory: AgentMemory, region_mask: np.ndarray) -> dict:
        """Tree-of-Thought：对同一区域并行生成多个解，选最优。"""
        branches = [
            {"name": "orthogonal",   "iou_threshold": 0.85, "description": "正交化为直角墙段"},
            {"name": "diagonal",     "iou_threshold": 0.70, "description": "保留斜向墙体设计意图"},
            {"name": "fine_grained", "iou_threshold": 0.90, "description": "更精细的收缩，保边界"},
        ]
        results = []
        for branch in branches:
            from vector_logic import vectorize_wall_mask
            from postprocess_config import VectorizationConfig
            cfg        = VectorizationConfig(iou_threshold=branch["iou_threshold"])
            wall_boxes = vectorize_wall_mask(region_mask, cfg)
            score      = self._score_branch(wall_boxes, region_mask)
            results.append({"branch": branch, "wall_boxes": wall_boxes, "score": score})

        best                = max(results, key=lambda x: x["score"])
        memory.tot_branches = results
        return best

    def _score_branch(self, wall_boxes, gt_mask) -> float:
        from vector_logic import compute_vectorization_iou
        iou       = compute_vectorization_iou(wall_boxes, gt_mask)
        closure   = self._check_room_closure(wall_boxes)
        coherence = self._check_coherence(wall_boxes)
        return 0.5 * iou + 0.3 * closure + 0.2 * coherence

    def go_explore_backtrack(self, memory: AgentMemory, target_step: str) -> bool:
        """回溯到指定检查点，用于 3D 布尔失败后重试不同策略。"""
        checkpoint = memory.checkpoints.get(target_step)
        if checkpoint:
            memory.current_mask = checkpoint["mask"]
            memory.best_iou     = checkpoint["best_iou"]
            return True
        return False

    # ──────────────────────────────────────────
    # 主循环（接入 ThoughtLogger + evaluation）
    # ──────────────────────────────────────────

    def run(
        self,
        image_path:  str,
        image_rgb:   Optional[np.ndarray] = None,
        gt_mask:     Optional[np.ndarray] = None,
        wall_boxes:  Optional[list]       = None,
        openings:    Optional[list]       = None,
    ) -> dict:
        """
        完整 ReAct 循环主入口。

        新增参数（用于结尾 evaluate + FailureRAG）：
            image_rgb  : 原图，语义评分用；None 时跳过语义维度
            gt_mask    : GT mask，IoU 计算用；None 时退化为自比较
            wall_boxes : 矢量化结果；None 时从 memory 取
            openings   : VLM 输出；None 时用空列表
        """
        memory     = AgentMemory(image_path=image_path)
        MAX_STEPS  = 10
        tl         = ThoughtLogger(image_path, self.log_dir)
        tl.start()

        for step_id in range(MAX_STEPS):
            situation = self._describe_situation(memory)
            thought   = self.think(memory, situation)
            obs       = self.act(thought, memory)

            if obs.metrics.get("iou_mask", 0) > memory.best_iou:
                memory.best_iou     = obs.metrics["iou_mask"]
                memory.current_mask = (
                    obs.raw_output.get("wall_mask") if obs.raw_output else None
                )

            reflection = self.reflect(obs, memory)

            step = AgentStep(
                step_id     = step_id,
                state       = AgentState.REFLECTING if reflection else AgentState.ACTING,
                thought     = thought,
                action      = Action(tool_name=thought.tool_choice, tool_args={}, thought=thought),
                observation = obs,
                timestamp   = time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            )
            memory.steps.append(step)

            # ── ThoughtLogger：记录每步 ──
            tl.log_step_from_agent_step(step)

            if obs.success and memory.best_iou >= 0.75 and thought.tool_choice == "generate_svg":
                break

            key = thought.tool_choice
            memory.retry_count[key] = memory.retry_count.get(key, 0) + 1
            if memory.retry_count.get(key, 0) >= 3:
                break

        # ── 综合评分 ──
        pred_mask   = memory.current_mask if memory.current_mask is not None \
                      else np.zeros((1, 1), dtype=np.uint8)
        _gt_mask    = gt_mask    if gt_mask    is not None else np.zeros_like(pred_mask)
        _wall_boxes = wall_boxes if wall_boxes is not None else []
        _openings   = openings   if openings   is not None else []
        _image_rgb  = image_rgb  if image_rgb  is not None \
                      else np.zeros((1, 1, 3), dtype=np.uint8)

        eval_result: EvalResult = evaluate(
            pred_mask   = pred_mask,
            gt_mask     = _gt_mask,
            wall_boxes  = _wall_boxes,
            openings    = _openings,
            image_rgb   = _image_rgb,
            cfg         = self.cfg,
            weights     = self.eval_weights,
            vlm_client  = self.vlm_client,
        )

        # ── ThoughtLogger：记录评分 + 写报告 ──
        tl.log_eval(eval_result.to_dict())
        log_dict      = tl.finish()
        log_json_path = os.path.join(
            self.log_dir, f"{Path(image_path).stem}_thought_log.json"
        )

        # ── FailureRAG：失败时存入错题集 ──
        if self.failure_rag and not eval_result.passed:
            self.failure_rag.add(
                image_name       = Path(image_path).stem,
                situation        = self._describe_situation(memory),
                eval_result      = eval_result.to_dict(),
                thought_log_path = log_json_path,
            )

        return self._finalize(memory, eval_result, log_json_path)

    # ──────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────

    def _describe_situation(self, memory: AgentMemory) -> str:
        if not memory.steps:
            return "初始状态，尚未执行任何步骤，需要对图纸做推理生成初始 wall mask"
        last = memory.steps[-1]
        return (
            f"已完成 {len(memory.steps)} 步，"
            f"上一步工具={last.action.tool_name}，"
            f"成功={last.observation.success}，"
            f"当前最优 IoU={memory.best_iou:.3f}，"
            f"失败原因={last.observation.failure_reason or '无'}"
        )

    def _evaluate(self, result: dict, memory: AgentMemory) -> dict:
        metrics = {}
        if result is None:
            return metrics
        if "wall_mask" in result:
            mask = result["wall_mask"]
            if memory.current_mask is not None and mask is not None:
                inter = (mask.astype(bool) & memory.current_mask.astype(bool)).sum()
                union = (mask.astype(bool) | memory.current_mask.astype(bool)).sum()
                metrics["iou_mask"] = float(inter / (union + 1e-8))
            else:
                metrics["iou_mask"] = 1.0
        if "boxes" in result:
            metrics["det_n"] = len(result["boxes"])
        return metrics

    def _finalize(
        self,
        memory:       AgentMemory,
        eval_result:  EvalResult,
        log_path:     str,
    ) -> dict:
        last_obs = memory.steps[-1].observation if memory.steps else None
        return {
            "image_path":   memory.image_path,
            "best_iou":     memory.best_iou,
            "n_steps":      len(memory.steps),
            "success":      last_obs.success if last_obs else False,
            "svg_path":     (
                last_obs.raw_output.get("svg_path")
                if last_obs and isinstance(last_obs.raw_output, dict)
                else None
            ),
            "eval":         eval_result.to_dict(),
            "thought_log":  log_path,
            "audit_log": [
                {
                    "step_id":   s.step_id,
                    "state":     s.state.value,
                    "tool":      s.action.tool_name,
                    "success":   s.observation.success,
                    "metrics":   s.observation.metrics,
                    "failure":   s.observation.failure_reason,
                    "timestamp": s.timestamp,
                }
                for s in memory.steps
            ],
        }


# ── 缺失的 import（放文件底部避免循环依赖）──
import os
from pathlib import Path
