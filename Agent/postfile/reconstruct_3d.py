"""
reconstruct_3d.py — 算法层：3D 重建（Section 2.5）

依赖 trimesh，可在没有 GPU 的纯 CPU 环境独立部署。
如果只需要 2D 矢量化，不需要 import 本文件。

对外接口（post_service.py / preview_service.py 调用）：
    match_openings_to_walls(wall_boxes, opening_boxes, opening_labels) -> List[dict]
    build_3d_model(wall_boxes, openings, cfg, output_path) -> trimesh.Scene | None
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from postprocess_config import (
    VectorizationConfig, VECT_CFG,
    WALL_HEIGHT, DOOR_HEIGHT, WINDOW_HEIGHT, WINDOW_SILL,
    PIXELS_PER_METER, FLOOR_THICKNESS,
)

logger = logging.getLogger(__name__)

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    logger.warning('trimesh 未安装，3D 重建不可用。安装: pip install trimesh')


# ══════════════════════════════════════════════════════════════
# 几何工具（复用 vector_logic 的 box 运算）
# ══════════════════════════════════════════════════════════════

def _box_intersection(b1: Tuple, b2: Tuple) -> float:
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    return max(0.0, float(ix2 - ix1)) * max(0.0, float(iy2 - iy1))


# ══════════════════════════════════════════════════════════════
# Section 2.5 Step 1：门窗匹配到对应墙体段
# ══════════════════════════════════════════════════════════════

def match_openings_to_walls(
    wall_boxes:      List[Tuple],
    opening_boxes:   np.ndarray,
    opening_labels:  np.ndarray,
) -> List[dict]:
    """
    论文 Section 2.5：
    把检测到的门窗 bbox 匹配到 overlap 最大的墙体段

    返回列表，每项格式：
    {
        'opening_box': (x1, y1, x2, y2),
        'type':        'door' | 'window',
        'wall_idx':    int,
        'wall_box':    (x1, y1, x2, y2),
    }
    """
    matched = []

    for box, label in zip(opening_boxes, opening_labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue

        best_wall_idx = -1
        best_overlap  = 0.0

        for j, wall in enumerate(wall_boxes):
            inter = _box_intersection((x1, y1, x2, y2), wall)
            if inter > best_overlap:
                best_overlap  = inter
                best_wall_idx = j

        if best_wall_idx >= 0:
            matched.append({
                'opening_box': (x1, y1, x2, y2),
                'type':        'door' if int(label) == 1 else 'window',
                'wall_idx':    best_wall_idx,
                'wall_box':    wall_boxes[best_wall_idx],
            })

    logger.debug(f'门窗匹配: {len(matched)} / {len(opening_boxes)} 个成功匹配')
    return matched


# ══════════════════════════════════════════════════════════════
# Section 2.5 Step 2-4：构建 3D 网格并裁剪门窗开口
# ══════════════════════════════════════════════════════════════

def build_3d_model(
    wall_boxes:   List[Tuple],
    openings:     List[dict],
    scale:        float = 1.0 / PIXELS_PER_METER,
    wall_height:  float = WALL_HEIGHT,
    door_height:  float = DOOR_HEIGHT,
    win_height:   float = WINDOW_HEIGHT,
    win_sill:     float = WINDOW_SILL,
    floor_thick:  float = FLOOR_THICKNESS,
    output_path:  Optional[str] = None,
) -> Optional[object]:
    """
    论文 Section 2.5：3D 重建
    用矩形 box primitives 构建墙体，布尔减法裁剪门窗开口

    参数：
        wall_boxes   : vectorize_wall_mask 输出的墙体 bbox 列表
        openings     : match_openings_to_walls 输出的门窗匹配列表
        scale        : 像素 → 米 的缩放比
        output_path  : 若不为 None，导出 GLB/OBJ 到该路径

    返回：
        trimesh.Scene 对象，或 None（trimesh 未安装时）
    """
    if not HAS_TRIMESH:
        logger.error('需要安装 trimesh: pip install trimesh')
        return None

    if not wall_boxes:
        logger.warning('wall_boxes 为空，跳过 3D 重建')
        return None

    scene = trimesh.Scene()

    # 按墙体索引建立门窗索引
    wall_openings: dict = {i: [] for i in range(len(wall_boxes))}
    for op in openings:
        wall_openings[op['wall_idx']].append(op)

    for wall_idx, wall in enumerate(wall_boxes):
        wx1, wy1, wx2, wy2 = wall
        x1_m = wx1 * scale; y1_m = wy1 * scale
        x2_m = wx2 * scale; y2_m = wy2 * scale
        width_m = abs(x2_m - x1_m)
        depth_m = abs(y2_m - y1_m)

        if width_m < 0.01 or depth_m < 0.01:
            logger.debug(f'跳过过薄的墙体 wall_{wall_idx}')
            continue

        # 创建完整墙体 box
        wall_mesh = trimesh.creation.box(extents=[width_m, depth_m, wall_height])
        cx = (x1_m + x2_m) / 2
        cy = (y1_m + y2_m) / 2
        wall_mesh.apply_translation([cx, cy, wall_height / 2])

        # 裁剪门窗开口（布尔减法）
        for op in wall_openings[wall_idx]:
            ox1, oy1, ox2, oy2 = op['opening_box']
            op_w = abs((ox2 - ox1) * scale)
            op_d = abs((oy2 - oy1) * scale)

            if op['type'] == 'door':
                op_h     = door_height
                op_z_bot = 0.0
            else:
                op_h     = win_height
                op_z_bot = win_sill

            opening_box = trimesh.creation.box(
                extents=[op_w + 0.01, op_d + 0.01, op_h + 0.01]
            )
            ocx = ((ox1 + ox2) / 2) * scale
            ocy = ((oy1 + oy2) / 2) * scale
            opening_box.apply_translation([ocx, ocy, op_z_bot + op_h / 2])

            try:
                wall_mesh = trimesh.boolean.difference(
                    [wall_mesh, opening_box],
                    engine='blender',
                )
            except Exception as e:
                logger.warning(f'wall_{wall_idx} 门窗布尔减法失败，保留完整墙体: {e}')

        wall_mesh.visual.face_colors = [200, 200, 200, 255]
        scene.add_geometry(wall_mesh, node_name=f'wall_{wall_idx}')

    # 添加地板
    all_x = [b[0] for b in wall_boxes] + [b[2] for b in wall_boxes]
    all_y = [b[1] for b in wall_boxes] + [b[3] for b in wall_boxes]
    fx1 = min(all_x) * scale; fx2 = max(all_x) * scale
    fy1 = min(all_y) * scale; fy2 = max(all_y) * scale

    floor = trimesh.creation.box(extents=[fx2 - fx1, fy2 - fy1, floor_thick])
    floor.apply_translation([(fx1 + fx2) / 2, (fy1 + fy2) / 2, -floor_thick / 2])
    floor.visual.face_colors = [180, 160, 140, 255]
    scene.add_geometry(floor, node_name='floor')

    # 导出文件
    if output_path:
        output_path = str(output_path)
        try:
            scene.export(output_path)
            logger.info(f'3D 模型已导出: {output_path}')
        except Exception as e:
            logger.error(f'3D 模型导出失败: {e}')

    return scene
