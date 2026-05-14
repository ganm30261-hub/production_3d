# failure_rag.py
"""
错题集向量库：将标注失败的案例存入 FAISS，
当 Agent 下次遇到类似图纸时，先检索"前车之鉴"，从而绕过已知的坑。

存储结构：
    {rag_dir}/
        failure_index.faiss      向量索引
        failure_meta.jsonl       每条记录的元数据（与向量索引一一对应）

记录内容（每条）：
    image_name    图片名
    situation     失败时的情况描述（用于向量化检索）
    eval_result   EvalResult.to_dict()
    failure_tags  自动提取的失败类型标签，方便过滤
    thought_log_path  对应推理报告路径
    timestamp

检索策略：
    把当前 situation 编码成向量，找最相似的 K 个失败案例，
    返回结构化的 few-shot 文本，直接插入 FloorplanAgent.think() 的 prompt。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import logger

# 可选依赖
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning('[!] faiss 未安装: pip install faiss-cpu')

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    logger.warning('[!] sentence-transformers 未安装: pip install sentence-transformers')


# ══════════════════════════════════════════════════════════════
# 失败标签自动提取
# ══════════════════════════════════════════════════════════════

def _extract_failure_tags(eval_result: dict) -> List[str]:
    """
    从 EvalResult.to_dict() 自动提取可读的失败类型标签。
    标签用于快速过滤和日志展示，不影响向量检索。
    """
    tags = []
    if eval_result.get("iou_pixel", 1.0) < 0.5:
        tags.append("low_iou")
    if eval_result.get("iou_pixel", 1.0) < 0.75:
        tags.append("medium_iou")
    if eval_result.get("c_topological", 1.0) < 0.4:
        tags.append("topology_broken")
    if eval_result.get("s_semantic", 1.0) < 0.5 and eval_result.get("s_semantic", -1.0) >= 0:
        tags.append("semantic_issue")

    issues = eval_result.get("details", {}).get("semantic", {}).get("issues", [])
    for issue in issues[:2]:
        # 把 VLM 的问题描述截短成标签
        short = issue[:30].replace(" ", "_").replace("，", "").replace("。", "")
        tags.append(f"sem:{short}")

    topo_rooms = eval_result.get("details", {}).get("topology", {}).get("n_valid_rooms", -1)
    if topo_rooms == 0:
        tags.append("no_closed_room")

    return tags if tags else ["unknown_failure"]


# ══════════════════════════════════════════════════════════════
# FailureRAG
# ══════════════════════════════════════════════════════════════

class FailureRAG:
    """
    错题集向量库。

    用法：
        rag = FailureRAG(rag_dir="path/to/rag")

        # 记录失败案例（pipeline 结尾调用）
        rag.add(
            image_name     = "office_B3",
            situation      = "wall mask 边界模糊，IoU=0.48，拓扑断裂",
            eval_result    = eval_result.to_dict(),
            thought_log_path = "/path/to/office_B3_thought_log.json",
        )

        # 检索相似案例（floorplan_agent.think() 里调用）
        few_shot_text = rag.retrieve_as_fewshot(
            situation = "当前 mask 覆盖率过低，检测不到房间",
            top_k     = 3,
        )
    """

    EMBED_DIM  = 384   # all-MiniLM-L6-v2 的输出维度
    INDEX_FILE = "failure_index.faiss"
    META_FILE  = "failure_meta.jsonl"

    def __init__(self, rag_dir: str):
        self.rag_dir    = rag_dir
        self.index_path = os.path.join(rag_dir, self.INDEX_FILE)
        self.meta_path  = os.path.join(rag_dir, self.META_FILE)

        os.makedirs(rag_dir, exist_ok=True)

        self._encoder = None   # 懒加载
        self._index   = None   # 懒加载
        self._meta:   List[dict] = []

        # 如果已有持久化文件，恢复
        if os.path.exists(self.meta_path):
            self._load_meta()
        if HAS_FAISS and os.path.exists(self.index_path):
            self._load_index()
        elif HAS_FAISS:
            self._init_index()

    # ── 公开 API ──

    def add(
        self,
        image_name:       str,
        situation:        str,
        eval_result:      dict,
        thought_log_path: Optional[str] = None,
    ) -> None:
        """
        把一条失败案例加入向量库和元数据文件。
        只存 passed=False 的案例；passed=True 自动跳过。
        """
        if eval_result.get("passed", True):
            return   # 不存成功案例

        if not self._check_deps():
            return

        tags    = _extract_failure_tags(eval_result)
        record  = {
            "image_name":       image_name,
            "situation":        situation,
            "eval_result":      eval_result,
            "failure_tags":     tags,
            "thought_log_path": thought_log_path or "",
            "timestamp":        time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        }

        # 向量化 situation（检索键）
        vec = self._encode(situation)

        # 加入 FAISS
        self._index.add(vec)

        # 追加元数据
        self._meta.append(record)
        with open(self.meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 持久化 FAISS 索引
        faiss.write_index(self._index, self.index_path)

        logger.info(
            f"[FailureRAG] 新增失败案例: {image_name}  "
            f"tags={tags}  "
            f"total={len(self._meta)}"
        )

    def retrieve(
        self,
        situation: str,
        top_k:     int = 3,
        tag_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        检索与当前 situation 最相似的 top_k 条失败案例。

        参数:
            situation  : 当前处理阶段描述
            top_k      : 返回条数
            tag_filter : 只返回包含该标签的记录（可选过滤）

        返回:
            List[dict]  元数据列表，按相似度降序
        """
        if not self._check_deps() or len(self._meta) == 0:
            return []

        vec       = self._encode(situation)
        k_search  = min(top_k * 3, len(self._meta))   # 多取一些再过滤
        scores, indices = self._index.search(vec, k_search)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._meta):
                continue
            record = self._meta[idx].copy()
            record["_similarity"] = float(score)
            if tag_filter and tag_filter not in record.get("failure_tags", []):
                continue
            results.append(record)
            if len(results) >= top_k:
                break

        return results

    def retrieve_as_fewshot(
        self,
        situation: str,
        top_k:     int = 3,
    ) -> str:
        """
        检索失败案例并格式化为 few-shot 文本，
        直接插入 FloorplanAgent.think() 的 prompt。

        返回示例：
            【前车之鉴 1】图片: office_B3 | 标签: low_iou, topology_broken
            情况: wall mask 边界模糊，IoU=0.48
            教训: IoU_pixel=0.41, C_topo=0.22, S_total=0.35
            建议: 降低 iou_threshold，先用 morphological_preprocessing 修补
            ──
            【前车之鉴 2】...
        """
        cases = self.retrieve(situation, top_k=top_k)

        if not cases:
            return "（错题集为空，暂无参考案例）"

        lines = []
        for i, c in enumerate(cases, 1):
            er    = c.get("eval_result", {})
            tags  = ", ".join(c.get("failure_tags", []))
            sim   = c.get("_similarity", 0.0)
            suggestion = _make_suggestion(er)

            lines += [
                f"【前车之鉴 {i}】图片: {c['image_name']}  "
                f"相似度: {sim:.2f}  标签: {tags}",
                f"  情况: {c['situation']}",
                f"  评分: IoU={er.get('iou_pixel',0):.3f}  "
                f"Topo={er.get('c_topological',0):.3f}  "
                f"Total={er.get('s_total',0):.3f}",
                f"  建议: {suggestion}",
                "──",
            ]

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._meta)

    def stats(self) -> dict:
        """返回错题集统计摘要。"""
        if not self._meta:
            return {"total": 0}
        all_tags: List[str] = []
        for r in self._meta:
            all_tags.extend(r.get("failure_tags", []))
        from collections import Counter
        tag_counts = dict(Counter(all_tags).most_common(10))
        scores = [r["eval_result"].get("s_total", 0) for r in self._meta]
        return {
            "total":      len(self._meta),
            "avg_score":  round(float(np.mean(scores)), 4) if scores else 0,
            "tag_counts": tag_counts,
        }

    # ── 内部工具 ──

    def _check_deps(self) -> bool:
        if not HAS_FAISS or not HAS_ST:
            logger.warning("FailureRAG: faiss 或 sentence-transformers 未安装，跳过")
            return False
        return True

    def _get_encoder(self) -> "SentenceTransformer":
        if self._encoder is None:
            self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
        return self._encoder

    def _encode(self, text: str) -> np.ndarray:
        """把文本编码成 (1, EMBED_DIM) float32 向量。"""
        vec = self._get_encoder().encode([text]).astype('float32')
        faiss.normalize_L2(vec)   # 归一化，使 IndexFlatIP = 余弦相似度
        return vec

    def _init_index(self) -> None:
        self._index = faiss.IndexFlatIP(self.EMBED_DIM)

    def _load_index(self) -> None:
        self._index = faiss.read_index(self.index_path)
        logger.info(f"[FailureRAG] 恢复索引: {self._index.ntotal} 条向量")

    def _load_meta(self) -> None:
        with open(self.meta_path, encoding="utf-8") as f:
            self._meta = [json.loads(line) for line in f if line.strip()]
        logger.info(f"[FailureRAG] 恢复元数据: {len(self._meta)} 条记录")


# ══════════════════════════════════════════════════════════════
# 建议生成（把 eval 结果转成人类可读的改进建议）
# ══════════════════════════════════════════════════════════════

def _make_suggestion(eval_result: dict) -> str:
    """根据各维度评分自动生成改进建议。"""
    suggestions = []

    iou   = eval_result.get("iou_pixel",     1.0)
    topo  = eval_result.get("c_topological", 1.0)
    sem   = eval_result.get("s_semantic",    1.0)
    issues = eval_result.get("details", {}).get("semantic", {}).get("issues", [])

    if iou < 0.5:
        suggestions.append("IoU 极低：检查模型是否在该图纸风格上过拟合，考虑降低推理阈值或增加数据增强")
    elif iou < 0.75:
        suggestions.append("IoU 偏低：降低 iou_threshold 并先运行 morphological_preprocessing 修补 mask")

    if topo < 0.4:
        suggestions.append("拓扑断裂：用 Closing 操作填补墙体缺口，或降低 shrink_iou_thresh 使墙段更连续")

    if 0 <= sem < 0.5:
        if issues:
            suggestions.append(f"语义问题: {issues[0]}；考虑重新运行 vlm_semantic_completion")
        else:
            suggestions.append("语义合理性低：检查门窗是否悬空于墙体之外")

    return "；".join(suggestions) if suggestions else "评分偏低但无明确诊断，建议人工复查"
