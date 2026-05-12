# tools/svg_export.py
import os
import time
from typing import List, Tuple

from config import logger, PseudoLabelConfig


def generate_svg(
    image_wh:    Tuple[int, int],
    wall_boxes:  List[Tuple],
    openings:    List[dict],
    cfg:         PseudoLabelConfig,
    output_path: str,
) -> str:
    """
    把 wall_boxes 和 openings 生成 CubiCasa 兼容的 SVG 标注文件。

    CubiCasa SVG 关键约定：
        <g id="Floor0">          包含所有楼层元素
        <rect class="Wall">      墙体
        <rect class="Door">      门
        <rect class="Window">    窗

    门窗额外携带两个 data 属性，方便后续解析或人工审查：
        data-confidence  : VLM 置信度
        data-wall-side   : 所在墙面朝向

    参数:
        image_wh    : (width, height) 原图像素尺寸，用于 SVG viewBox
        wall_boxes  : List[Tuple[x1,y1,x2,y2]]，Shrinking 矢量化输出
        openings    : List[dict]，VLM 语义补全输出，每项含 type/bbox/wall_side/confidence
        cfg         : PseudoLabelConfig（目前仅用于日志，预留扩展）
        output_path : SVG 文件写入路径

    返回:
        svg_content : str，写入文件的完整 SVG 文本
    """
    W, H  = image_wh
    lines = []

    # ── SVG 头部 ──
    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
    )
    lines.append('  <g id="Floor0">')

    # ── 墙体 ──
    lines.append('    <!-- Walls -->')
    for i, b in enumerate(wall_boxes):
        x1, y1, x2, y2 = [int(v) for v in b]
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        lines.append(
            f'    <rect id="wall_{i}" class="Wall" '
            f'x="{x1}" y="{y1}" width="{bw}" height="{bh}" '
            f'fill="#333333" stroke="none"/>'
        )

    # ── 门 ──
    lines.append('    <!-- Doors -->')
    door_count = 0
    for op in openings:
        if op.get('type') != 'door':
            continue
        x1, y1, x2, y2 = [int(v) for v in op['bbox']]
        bw   = max(x2 - x1, 1)
        bh   = max(y2 - y1, 1)
        conf = op.get('confidence', 1.0)
        side = op.get('wall_side', 'unknown')
        lines.append(
            f'    <rect id="door_{door_count}" class="Door" '
            f'x="{x1}" y="{y1}" width="{bw}" height="{bh}" '
            f'fill="#8B4513" stroke="none" '
            f'data-confidence="{conf:.2f}" '
            f'data-wall-side="{side}"/>'
        )
        door_count += 1

    # ── 窗 ──
    lines.append('    <!-- Windows -->')
    win_count = 0
    for op in openings:
        if op.get('type') != 'window':
            continue
        x1, y1, x2, y2 = [int(v) for v in op['bbox']]
        bw   = max(x2 - x1, 1)
        bh   = max(y2 - y1, 1)
        conf = op.get('confidence', 1.0)
        side = op.get('wall_side', 'unknown')
        lines.append(
            f'    <rect id="window_{win_count}" class="Window" '
            f'x="{x1}" y="{y1}" width="{bw}" height="{bh}" '
            f'fill="#87CEEB" stroke="#4169E1" stroke-width="1" '
            f'data-confidence="{conf:.2f}" '
            f'data-wall-side="{side}"/>'
        )
        win_count += 1

    # ── 元数据注释 ──
    lines.append(
        f'    <!-- Meta: walls={len(wall_boxes)} '
        f'doors={door_count} windows={win_count} '
        f'generated={time.strftime("%Y-%m-%dT%H:%M:%SZ")} -->'
    )
    lines.append('  </g>')
    lines.append('</svg>')

    svg_content = '\n'.join(lines)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)

    logger.info(
        f'SVG 生成完成: {output_path}  '
        f'walls={len(wall_boxes)}  '
        f'doors={door_count}  '
        f'windows={win_count}'
    )
    return svg_content
