# thought_logger.py
"""
推理日志模块：把每张图的标注过程从"黑盒"变成可读的推理报告。

每张图生成两个文件：
    {name}_thought_log.json   机器可读，用于错题集 RAG 和后续分析
    {name}_thought_log.md     人类可读，方便 debug 和展示

日志结构：
    header        图片基本信息
    steps[]       每个 ReAct 步骤的 Thought / Action / Observation
    eval_result   最终综合评分
    conclusion    自动生成的一句话结论（passed / failed + 原因）
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from config import logger


# ══════════════════════════════════════════════════════════════
# 日志数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class StepLog:
    """单个 ReAct 步骤的完整记录。"""
    step_id:        int
    timestamp:      str
    state:          str

    # Thought
    reasoning:      str
    plan:           List[str]
    tool_choice:    str
    confidence:     float

    # Observation
    success:        bool
    metrics:        dict
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "step_id":        self.step_id,
            "timestamp":      self.timestamp,
            "state":          self.state,
            "thought": {
                "reasoning":   self.reasoning,
                "plan":        self.plan,
                "tool_choice": self.tool_choice,
                "confidence":  self.confidence,
            },
            "observation": {
                "success":        self.success,
                "metrics":        self.metrics,
                "failure_reason": self.failure_reason,
            },
        }

    def to_md(self) -> str:
        status  = "✓" if self.success else "✗"
        metrics = "  ".join(f"`{k}={v:.3f}`" if isinstance(v, float)
                            else f"`{k}={v}`"
                            for k, v in self.metrics.items())
        lines = [
            f"### Step {self.step_id}  [{self.state}]  {self.timestamp}",
            f"",
            f"**Thought**",
            f"> {self.reasoning}",
            f"",
            f"- 计划: {' → '.join(self.plan)}",
            f"- 工具: `{self.tool_choice}`  置信度: {self.confidence:.2f}",
            f"",
            f"**Observation** {status}",
        ]
        if metrics:
            lines.append(f"- 指标: {metrics}")
        if self.failure_reason:
            lines.append(f"- 失败原因: _{self.failure_reason}_")
        lines.append("")
        return "\n".join(lines)


@dataclass
class ThoughtLog:
    """一张图完整的推理日志。"""
    image_path:   str
    image_name:   str
    start_time:   str
    end_time:     str   = ""
    elapsed_s:    float = 0.0
    steps:        List[StepLog] = field(default_factory=list)
    eval_result:  dict          = field(default_factory=dict)
    conclusion:   str           = ""

    def to_dict(self) -> dict:
        return {
            "image_path":  self.image_path,
            "image_name":  self.image_name,
            "start_time":  self.start_time,
            "end_time":    self.end_time,
            "elapsed_s":   round(self.elapsed_s, 2),
            "n_steps":     len(self.steps),
            "steps":       [s.to_dict() for s in self.steps],
            "eval_result": self.eval_result,
            "conclusion":  self.conclusion,
        }

    def to_md(self) -> str:
        passed  = self.eval_result.get("passed", False)
        s_total = self.eval_result.get("s_total", 0.0)
        badge   = "🟢 PASSED" if passed else "🔴 FAILED"

        header = [
            f"# 推理报告：{self.image_name}",
            f"",
            f"| 项目 | 值 |",
            f"|------|----|",
            f"| 图片 | `{self.image_path}` |",
            f"| 开始 | {self.start_time} |",
            f"| 耗时 | {self.elapsed_s:.1f}s |",
            f"| 步数 | {len(self.steps)} |",
            f"| 总分 | **{s_total:.3f}** |",
            f"| 结果 | {badge} |",
            f"",
            f"## 评分详情",
            f"",
        ]

        # 评分表格
        if self.eval_result:
            w = self.eval_result.get("weights", {})
            header += [
                f"| 维度 | 分数 | 权重 |",
                f"|------|------|------|",
                f"| IoU_pixel     | {self.eval_result.get('iou_pixel', 0):.3f} | {w.get('w1_iou', 0):.2f} |",
                f"| C_topological | {self.eval_result.get('c_topological', 0):.3f} | {w.get('w2_topology', 0):.2f} |",
                f"| S_semantic    | {self.eval_result.get('s_semantic', 0):.3f} | {w.get('w3_semantic', 0):.2f} |",
                f"| **S_total**   | **{s_total:.3f}** | — |",
                f"",
            ]

        header += [
            f"**结论：** {self.conclusion}",
            f"",
            f"---",
            f"",
            f"## 推理步骤",
            f"",
        ]

        step_md = [s.to_md() for s in self.steps]

        return "\n".join(header) + "\n".join(step_md)


# ══════════════════════════════════════════════════════════════
# ThoughtLogger
# ══════════════════════════════════════════════════════════════

class ThoughtLogger:
    """
    管理单张图的推理日志生命周期：开始 → 记录步骤 → 记录评分 → 写文件。

    典型用法（pipeline 里）：
        tl = ThoughtLogger(image_path, output_dir)
        tl.start()
        tl.log_step(step_id, state, thought, observation)
        ...
        tl.log_eval(eval_result.to_dict())
        tl.finish()   # 写 JSON + MD 文件
    """

    def __init__(self, image_path: str, output_dir: str):
        self.image_path = image_path
        self.image_name = Path(image_path).stem
        self.output_dir = output_dir
        self._log: Optional[ThoughtLog] = None
        self._t0: float = 0.0

    # ── 生命周期 ──

    def start(self) -> None:
        """开始一张图的记录，重置内部状态。"""
        self._t0  = time.time()
        self._log = ThoughtLog(
            image_path = self.image_path,
            image_name = self.image_name,
            start_time = time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        )
        logger.debug(f"[ThoughtLogger] 开始记录: {self.image_name}")

    def log_step(
        self,
        step_id:        int,
        state:          str,
        reasoning:      str,
        plan:           List[str],
        tool_choice:    str,
        confidence:     float,
        success:        bool,
        metrics:        dict,
        failure_reason: Optional[str] = None,
    ) -> None:
        """记录一个 ReAct 步骤。对应 agent_types.AgentStep。"""
        self._ensure_started()
        self._log.steps.append(StepLog(
            step_id        = step_id,
            timestamp      = time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            state          = state,
            reasoning      = reasoning,
            plan           = plan,
            tool_choice    = tool_choice,
            confidence     = confidence,
            success        = success,
            metrics        = metrics,
            failure_reason = failure_reason,
        ))

    def log_step_from_agent_step(self, agent_step) -> None:
        """
        直接从 agent_types.AgentStep 对象记录，
        方便在 floorplan_agent.run() 里无缝接入。
        """
        t  = agent_step.thought
        o  = agent_step.observation
        self.log_step(
            step_id        = agent_step.step_id,
            state          = agent_step.state.value,
            reasoning      = t.reasoning,
            plan           = t.plan,
            tool_choice    = t.tool_choice,
            confidence     = t.confidence,
            success        = o.success,
            metrics        = o.metrics,
            failure_reason = o.failure_reason,
        )

    def log_eval(self, eval_dict: dict) -> None:
        """记录 EvalResult.to_dict() 的输出。"""
        self._ensure_started()
        self._log.eval_result = eval_dict
        self._log.conclusion  = self._make_conclusion(eval_dict)

    def finish(self) -> dict:
        """
        结束记录，写 JSON + MD 文件，返回 ThoughtLog.to_dict()。
        即使 log_eval 未被调用也能安全写出（结论为空）。
        """
        self._ensure_started()
        self._log.end_time  = time.strftime('%Y-%m-%dT%H:%M:%SZ')
        self._log.elapsed_s = time.time() - self._t0

        os.makedirs(self.output_dir, exist_ok=True)
        json_path = os.path.join(self.output_dir, f"{self.image_name}_thought_log.json")
        md_path   = os.path.join(self.output_dir, f"{self.image_name}_thought_log.md")

        log_dict = self._log.to_dict()

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(log_dict, f, indent=2, ensure_ascii=False)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._log.to_md())

        logger.info(
            f"[ThoughtLogger] 报告已写出: "
            f"{Path(json_path).name}  "
            f"steps={len(self._log.steps)}  "
            f"elapsed={self._log.elapsed_s:.1f}s"
        )
        return log_dict

    # ── 训练阶段批量日志（train_one_version 里调用）──

    @staticmethod
    def log_epoch_summary(
        output_dir: str,
        version:    str,
        epoch:      int,
        metrics:    dict,
    ) -> None:
        """
        在训练主循环里记录每个 epoch 的摘要，追加写入同一个 JSONL 文件。
        不影响单图推理日志，独立文件以防混淆。

        文件：{output_dir}/training_thought_log.jsonl
        """
        os.makedirs(output_dir, exist_ok=True)
        record = {
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "version":   version,
            "epoch":     epoch,
            "metrics":   {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in metrics.items()},
            "note": _epoch_note(metrics),
        }
        log_path = os.path.join(output_dir, "training_thought_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── 内部工具 ──

    def _ensure_started(self) -> None:
        if self._log is None:
            raise RuntimeError(
                "ThoughtLogger.start() 必须在 log_step / log_eval / finish 之前调用"
            )

    @staticmethod
    def _make_conclusion(eval_dict: dict) -> str:
        """根据评分结果自动生成一句话结论。"""
        s_total  = eval_dict.get("s_total", 0.0)
        passed   = eval_dict.get("passed",  False)
        issues   = eval_dict.get("details", {}).get("semantic", {}).get("issues", [])
        topo_n   = eval_dict.get("details", {}).get("topology", {}).get("n_valid_rooms", 0)

        if passed:
            return f"标注质量合格（S_total={s_total:.3f}），检测到 {topo_n} 个有效房间。"

        reasons = []
        if eval_dict.get("iou_pixel", 1.0) < 0.6:
            reasons.append("像素 IoU 过低（墙体分割不准）")
        if eval_dict.get("c_topological", 1.0) < 0.4:
            reasons.append("拓扑闭合性差（墙体未围成封闭房间）")
        if issues:
            reasons.append(f"语义问题: {issues[0]}")

        reason_str = "；".join(reasons) if reasons else "综合评分不足"
        return f"标注质量不合格（S_total={s_total:.3f}）：{reason_str}。"


def _epoch_note(metrics: dict) -> str:
    """为 epoch 摘要生成简短的文字描述，方便人工快速扫描日志。"""
    iou  = metrics.get("val_iou", 0.0)
    loss = metrics.get("train_loss", 0.0)
    if iou >= 0.80:
        return f"优秀  val_iou={iou:.3f}"
    if iou >= 0.70:
        return f"合格  val_iou={iou:.3f}"
    if iou >= 0.50:
        return f"改善中  val_iou={iou:.3f}  loss={loss:.4f}"
    return f"早期训练  val_iou={iou:.3f}  loss={loss:.4f}"
