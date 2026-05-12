# tools/vlm_completion.py
import base64
import io
import json

import numpy as np
from PIL import Image as PILImage

from config import logger, PseudoLabelConfig

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    logger.warning('[!] anthropic 未安装: pip install anthropic')


# ══════════════════════════════════════════════════════════════
# Prompt 模板
# ══════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
你是一位专业的建筑图纸识别工程师。
你将收到一张平面图图片和一张对应的墙体分割 mask。
你的任务是识别图中所有的门和窗，并输出结构化的 JSON 数据。
输出必须是合法的 JSON，不包含任何 Markdown 标记或解释文字。\
"""

_USER_PROMPT = """\
请仔细观察这张平面图（第一张图）和对应的墙体 mask（第二张图）。

识别所有门和窗，对每个开口输出：
  - type: "door" 或 "window"
  - bbox: [x1, y1, x2, y2]（像素坐标，左上角到右下角）
  - wall_side: "north"/"south"/"east"/"west"（开口所在墙面朝向，不确定填 "unknown"）
  - estimated_width_m: 估算宽度（米），门通常 0.8~1.2m，窗通常 0.6~2.4m
  - confidence: 0.0~1.0（你对这个识别的置信度）

同时输出整体信息：
  - n_rooms: 估计的房间数量
  - floor_area_m2: 估计的建筑面积（平方米），不确定填 null

返回格式（严格 JSON）：
{
  "openings": [
    {"type": "door", "bbox": [x1,y1,x2,y2], "wall_side": "north",
     "estimated_width_m": 0.9, "confidence": 0.95},
    ...
  ],
  "n_rooms": 3,
  "floor_area_m2": 85.0
}\
"""


# ══════════════════════════════════════════════════════════════
# 内部工具函数
# ══════════════════════════════════════════════════════════════

def _to_base64(image_rgb: np.ndarray) -> str:
    """RGB ndarray → PNG base64 字符串。"""
    pil = PILImage.fromarray(image_rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return base64.standard_b64encode(buf.getvalue()).decode()


def _overlay_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    把 mask 以绿色半透明叠加到原图，让 VLM 同时看到图像纹理
    （识别门窗符号）和墙体位置（定位开口所在墙）。
    """
    vis = image_rgb.copy()
    vis[mask == 1] = (
        vis[mask == 1] * 0.6 + np.array([0, 200, 0]) * 0.4
    ).astype(np.uint8)
    return vis


def _parse_json(text: str) -> dict:
    """
    解析 VLM 返回的 JSON 文本。
    兼容 VLM 偶尔在 JSON 外面包 ```json ... ``` 的情况。
    解析失败时返回带 _parse_error 标记的空结果，不抛异常。
    """
    text = text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        # 去掉首行的 ```json 和尾行的 ```
        end = -1 if lines[-1].strip() == '```' else len(lines)
        text = '\n'.join(lines[1:end])

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f'VLM JSON 解析失败，原始文本前200字: {text[:200]}')
        return {
            'openings':      [],
            'n_rooms':       None,
            'floor_area_m2': None,
            '_parse_error':  True,
        }


# ══════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════

def vlm_semantic_completion(
    image_rgb:    np.ndarray,
    refined_mask: np.ndarray,
    cfg:          PseudoLabelConfig,
    client,
) -> dict:
    """
    Step 3：VLM 语义补全，识别门窗类型 / 朝向 / 尺寸。

    输入:
        image_rgb    : (H, W, 3) uint8 RGB 原图
        refined_mask : (H, W)   uint8  0/1，SAM2 精化后的 wall mask
        cfg          : PseudoLabelConfig
        client       : anthropic.Anthropic() 实例

    输出 dict:
        {
            "openings": [
                {
                    "type": "door" | "window",
                    "bbox": [x1, y1, x2, y2],
                    "wall_side": "north" | "south" | "east" | "west" | "unknown",
                    "estimated_width_m": float | None,
                    "confidence": float,
                }
            ],
            "n_rooms": int | None,
            "floor_area_m2": float | None,
            "_tokens": {"input": int, "output": int},   # token 用量，方便成本核算
        }
    """
    if not HAS_ANTHROPIC:
        logger.error('anthropic 未安装: pip install anthropic')
        return {'openings': [], '_error': 'anthropic not installed'}

    # 第一张：原图；第二张：mask 叠加可视化
    img_b64  = _to_base64(image_rgb)
    mask_b64 = _to_base64(_overlay_mask(image_rgb, refined_mask))

    response = client.messages.create(
        model      = cfg.vlm_model,
        max_tokens = cfg.vlm_max_tokens,
        system     = _SYSTEM_PROMPT,
        messages   = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'source': {
                    'type': 'base64', 'media_type': 'image/png', 'data': img_b64,
                }},
                {'type': 'image', 'source': {
                    'type': 'base64', 'media_type': 'image/png', 'data': mask_b64,
                }},
                {'type': 'text', 'text': _USER_PROMPT},
            ],
        }],
    )

    result = _parse_json(response.content[0].text)
    result['_tokens'] = {
        'input':  response.usage.input_tokens,
        'output': response.usage.output_tokens,
    }
    return result
