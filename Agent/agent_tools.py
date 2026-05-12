# agent_tools.py
from dataclasses import dataclass
from typing import Callable, List

# ── 从 tools/ 导入具体实现 ──
from pipeline import run_inference
from tools.sam2_refine import refine_mask_with_sam2
from tools.vlm_completion import vlm_semantic_completion
from tools.svg_export import generate_svg
from tools.sam2_refine import load_sam2_predictor   # noqa: F401，供外部直接 import


# ══════════════════════════════════════════════════════════════
# Tool 数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class Tool:
    name:          str
    description:   str    # 供 ToolRAG 向量检索的自然语言描述
    function:      Callable
    input_schema:  dict   # 参数说明（文档用途）
    output_schema: dict   # 输出说明（文档用途）


# ══════════════════════════════════════════════════════════════
# Tool 注册表
# ══════════════════════════════════════════════════════════════

TOOL_REGISTRY: dict[str, Tool] = {

    "run_inference": Tool(
        name        = "run_inference",
        description = "用 DINOv2+LoRA 模型对图片做滑动窗口推理，生成初始 wall mask 和门窗 bbox",
        function    = run_inference,
        input_schema  = {
            "image": "np.ndarray RGB (H,W,3)",
            "model": "DINOv2LoRAModel",
            "cfg":   "PseudoLabelConfig（可选，默认 CFG）",
        },
        output_schema = {
            "wall_mask": "np.ndarray uint8 (H,W)  0/1",
            "boxes":     "np.ndarray float32 (N,4) [x1,y1,x2,y2]",
            "scores":    "np.ndarray float32 (N,)",
            "labels":    "np.ndarray int64 (N,)  1=door 2=window",
        },
    ),

    "refine_mask_with_sam2": Tool(
        name        = "refine_mask_with_sam2",
        description = "用 SAM2 Point Prompt 精化粗糙的 wall mask，改善边界清晰度",
        function    = refine_mask_with_sam2,
        input_schema  = {
            "image_rgb":    "np.ndarray RGB (H,W,3)",
            "initial_mask": "np.ndarray uint8 (H,W)",
            "sam2_predictor": "SAM2ImagePredictor",
            "cfg":          "PseudoLabelConfig",
        },
        output_schema = {
            "refined_mask": "np.ndarray uint8 (H,W)  0/1",
        },
    ),

    "vectorize_wall_mask": Tool(
        name        = "vectorize_wall_mask",
        description = "形态学预处理 + Hough 角度检测 + Shrinking 算法，把 mask 转成矩形 bbox 列表",
        function    = None,   # 实现在 vector_logic.py（CubiCasa 仓库），运行时动态绑定
        input_schema  = {
            "wall_mask":     "np.ndarray uint8 (H,W)",
            "iou_threshold": "float，默认 0.85",
        },
        output_schema = {
            "wall_boxes": "List[Tuple[int,int,int,int]]",
        },
    ),

    "morphological_preprocessing": Tool(
        name        = "morphological_preprocessing",
        description = "形态学 Opening 去噪 + Closing 填洞，用于 mask 质量差时的修补",
        function    = None,   # 实现在 vector_logic.py，运行时动态绑定
        input_schema  = {
            "wall_mask":   "np.ndarray uint8 (H,W)",
            "open_size":   "int",
            "close_size":  "int",
        },
        output_schema = {
            "cleaned_mask": "np.ndarray uint8 (H,W)",
        },
    ),

    "vlm_semantic_completion": Tool(
        name        = "vlm_semantic_completion",
        description = "把图片和 mask 发给 Claude VLM，识别门窗类型、朝向、尺寸",
        function    = vlm_semantic_completion,
        input_schema  = {
            "image_rgb":    "np.ndarray RGB (H,W,3)",
            "refined_mask": "np.ndarray uint8 (H,W)",
            "cfg":          "PseudoLabelConfig",
            "client":       "anthropic.Anthropic()",
        },
        output_schema = {
            "openings":      "List[dict]  type/bbox/wall_side/estimated_width_m/confidence",
            "n_rooms":       "int | None",
            "floor_area_m2": "float | None",
            "_tokens":       "dict  input/output token 用量",
        },
    ),

    "generate_svg": Tool(
        name        = "generate_svg",
        description = "把 wall_boxes 和 openings 生成 CubiCasa 兼容的 SVG 标注文件",
        function    = generate_svg,
        input_schema  = {
            "image_wh":    "Tuple[int,int]  (width, height)",
            "wall_boxes":  "List[Tuple]",
            "openings":    "List[dict]",
            "cfg":         "PseudoLabelConfig",
            "output_path": "str",
        },
        output_schema = {
            "svg_content": "str  写入文件的完整 SVG 文本",
        },
    ),
}


# ══════════════════════════════════════════════════════════════
# 运行时动态绑定 vector_logic 工具
# ══════════════════════════════════════════════════════════════

def _bind_vector_logic() -> None:
    """尝试绑定 vector_logic 里的函数，失败时保持 function=None 并 warning。"""
    try:
        from postfile.vector_logic import vectorize_wall_mask, morphological_preprocessing
        TOOL_REGISTRY["vectorize_wall_mask"].function        = vectorize_wall_mask
        TOOL_REGISTRY["morphological_preprocessing"].function = morphological_preprocessing
    except ImportError:
        from config import logger
        logger.warning(
            'vector_logic 未找到，vectorize_wall_mask / morphological_preprocessing '
            '的 function 暂为 None。确认 CubiCasa 仓库路径已加入 sys.path。'
        )

_bind_vector_logic()


# ══════════════════════════════════════════════════════════════
# ToolRAG：基于 FAISS 的工具语义检索
# ══════════════════════════════════════════════════════════════

class ToolRAG:
    """
    根据当前情况的自然语言描述，用向量相似度检索最合适的工具。
    索引在首次实例化时构建，之后每次 retrieve 只做一次 encode + search。
    """

    def __init__(self):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                f'ToolRAG 依赖未安装: {e}\n'
                f'pip install faiss-cpu sentence-transformers'
            ) from e

        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        names  = list(TOOL_REGISTRY.keys())
        descs  = [TOOL_REGISTRY[n].description for n in names]
        embeds = self.encoder.encode(descs).astype('float32')

        self.index = faiss.IndexFlatIP(embeds.shape[1])
        self.index.add(embeds)
        self.names = names

    def retrieve(self, situation: str, top_k: int = 3) -> List[str]:
        """
        根据当前情况描述检索最合适的 top_k 个工具名。

        参数:
            situation : 当前处理阶段的自然语言描述
                        例："wall mask 边界模糊，需要精化"
            top_k     : 返回的工具数量
        返回:
            List[str]  工具名列表，按相似度降序排列
        """
        q      = self.encoder.encode([situation]).astype('float32')
        _, idx = self.index.search(q, top_k)
        return [self.names[i] for i in idx[0]]
