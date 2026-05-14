# contracts.py
"""
数据契约：支柱2 —— 锁死所有 Agent 之间的 I/O 格式

三类契约：
    LabelContract       标注 Agent 的输出（CubiCasa 兼容 SVG + JSON）
    TrainingContract    训练 Agent 期待的文件夹结构和元数据
    ReconstructContract 3D 重建 Agent 接收的矢量中间件格式

原则：
    - 格式现在就锁死，即使现在用 mock 数据填充
    - 所有 Agent 只通过这些契约对象通信，不直接传 dict
    - validate() 方法在每次交接时自动校验，不合格抛 ValidationError
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple

from config import logger


# ══════════════════════════════════════════════════════════════
# 基础类型
# ══════════════════════════════════════════════════════════════

class ValidationError(Exception):
    """数据不符合契约格式，记录 Warning 并跳过该图，不阻断整体流程。"""
    pass


@dataclass
class BBox:
    """像素坐标矩形框，左上角到右下角。"""
    x1: int
    y1: int
    x2: int
    y2: int

    def validate(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValidationError(
                f'BBox 无效: ({self.x1},{self.y1})-({self.x2},{self.y2})  '
                f'宽={self.x2-self.x1} 高={self.y2-self.y1}'
            )

    def to_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, lst: List) -> BBox:
        return cls(*[int(v) for v in lst])

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


# ══════════════════════════════════════════════════════════════
# 契约1：标注输出（LabelContract）
# ══════════════════════════════════════════════════════════════

@dataclass
class WallBox:
    """单段墙体的矢量表示。"""
    bbox:        BBox
    orientation: str   = 'unknown'   # 'horizontal' | 'vertical' | 'diagonal' | 'unknown'
    confidence:  float = 1.0

    def validate(self):
        self.bbox.validate()
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError(f'WallBox.confidence 超出范围: {self.confidence}')
        if self.orientation not in ('horizontal', 'vertical', 'diagonal', 'unknown'):
            raise ValidationError(f'WallBox.orientation 无效: {self.orientation}')


@dataclass
class Opening:
    """门或窗的标注。"""
    type:               str           # 'door' | 'window'
    bbox:               BBox
    wall_side:          str   = 'unknown'   # 'north'|'south'|'east'|'west'|'unknown'
    estimated_width_m:  Optional[float] = None
    confidence:         float = 1.0

    VALID_TYPES      = ('door', 'window')
    VALID_WALL_SIDES = ('north', 'south', 'east', 'west', 'unknown')

    def validate(self):
        if self.type not in self.VALID_TYPES:
            raise ValidationError(f'Opening.type 无效: {self.type}')
        self.bbox.validate()
        if self.wall_side not in self.VALID_WALL_SIDES:
            raise ValidationError(f'Opening.wall_side 无效: {self.wall_side}')
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError(f'Opening.confidence 超出范围: {self.confidence}')
        if self.estimated_width_m is not None and self.estimated_width_m <= 0:
            raise ValidationError(f'Opening.estimated_width_m 必须为正数')


@dataclass
class LabelContract:
    """
    标注 Agent 的输出契约。
    100% 对应 CubiCasa SVG 的语义结构。

    必填字段（validate 强制检查）：
        image_path, image_wh, wall_boxes

    可选字段（允许为空列表）：
        openings, rooms

    输出文件：
        svg_path    → CubiCasa 兼容的 SVG 标注文件
        meta_path   → 对应的 JSON 元数据（训练时读）
    """
    image_path:   str
    image_wh:     Tuple[int, int]        # (width, height)
    wall_boxes:   List[WallBox]          = field(default_factory=list)
    openings:     List[Opening]          = field(default_factory=list)
    n_rooms:      Optional[int]          = None
    floor_area_m2: Optional[float]       = None
    svg_path:     Optional[str]          = None
    meta_path:    Optional[str]          = None
    eval_scores:  Dict[str, float]       = field(default_factory=dict)
    source:       str                    = 'model'  # 'model' | 'vlm' | 'manual'

    def validate(self):
        """校验契约完整性，不合格抛 ValidationError。"""
        if not self.image_path:
            raise ValidationError('LabelContract.image_path 不能为空')
        if len(self.image_wh) != 2 or any(v <= 0 for v in self.image_wh):
            raise ValidationError(f'LabelContract.image_wh 无效: {self.image_wh}')
        if len(self.wall_boxes) == 0:
            raise ValidationError('LabelContract.wall_boxes 为空，标注无效')

        for i, w in enumerate(self.wall_boxes):
            try:
                w.validate()
            except ValidationError as e:
                raise ValidationError(f'wall_boxes[{i}] 校验失败: {e}')

        for i, o in enumerate(self.openings):
            try:
                o.validate()
            except ValidationError as e:
                raise ValidationError(f'openings[{i}] 校验失败: {e}')

    def to_dict(self) -> dict:
        return {
            'image_path':    self.image_path,
            'image_wh':      list(self.image_wh),
            'wall_boxes': [
                {'bbox': w.bbox.to_list(), 'orientation': w.orientation,
                 'confidence': w.confidence}
                for w in self.wall_boxes
            ],
            'openings': [
                {'type': o.type, 'bbox': o.bbox.to_list(),
                 'wall_side': o.wall_side,
                 'estimated_width_m': o.estimated_width_m,
                 'confidence': o.confidence}
                for o in self.openings
            ],
            'n_rooms':      self.n_rooms,
            'floor_area_m2': self.floor_area_m2,
            'svg_path':     self.svg_path,
            'meta_path':    self.meta_path,
            'eval_scores':  self.eval_scores,
            'source':       self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LabelContract:
        wall_boxes = [
            WallBox(
                bbox        = BBox.from_list(w['bbox']),
                orientation = w.get('orientation', 'unknown'),
                confidence  = w.get('confidence', 1.0),
            )
            for w in d.get('wall_boxes', [])
        ]
        openings = [
            Opening(
                type              = o['type'],
                bbox              = BBox.from_list(o['bbox']),
                wall_side         = o.get('wall_side', 'unknown'),
                estimated_width_m = o.get('estimated_width_m'),
                confidence        = o.get('confidence', 1.0),
            )
            for o in d.get('openings', [])
        ]
        return cls(
            image_path    = d['image_path'],
            image_wh      = tuple(d['image_wh']),
            wall_boxes    = wall_boxes,
            openings      = openings,
            n_rooms       = d.get('n_rooms'),
            floor_area_m2 = d.get('floor_area_m2'),
            svg_path      = d.get('svg_path'),
            meta_path     = d.get('meta_path'),
            eval_scores   = d.get('eval_scores', {}),
            source        = d.get('source', 'model'),
        )

    def save_meta(self, path: str) -> str:
        """把契约序列化为 JSON，保存到 path。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        self.meta_path = path
        return path

    @classmethod
    def load_meta(cls, path: str) -> LabelContract:
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    # ── Mock（骨架期用）──

    @classmethod
    def mock(cls, image_path: str, image_wh: Tuple[int, int] = (512, 512)) -> LabelContract:
        """
        生成假数据，格式与真实输出完全一致。
        骨架期用来跑通流程，等模型训好直接换掉。
        """
        W, H = image_wh
        return cls(
            image_path = image_path,
            image_wh   = image_wh,
            wall_boxes = [
                WallBox(BBox(0,   0,   W,   10),  orientation='horizontal'),
                WallBox(BBox(0,   H-10, W,  H),   orientation='horizontal'),
                WallBox(BBox(0,   0,   10,  H),   orientation='vertical'),
                WallBox(BBox(W-10,0,   W,   H),   orientation='vertical'),
            ],
            openings = [
                Opening('door',   BBox(W//2-30, 0, W//2+30, 10),  wall_side='north'),
                Opening('window', BBox(10, H//3, 10, H//3+60),    wall_side='west'),
            ],
            n_rooms      = 1,
            floor_area_m2 = round(W * H / 2500, 1),
            source       = 'mock',
        )


# ══════════════════════════════════════════════════════════════
# 契约2：训练元数据（TrainingContract）
# ══════════════════════════════════════════════════════════════

@dataclass
class TrainingContract:
    """
    训练 Agent 期待的输入格式。

    文件夹结构约定：
        {train_root}/
            images/   ← 原始图片（PNG）
            labels/   ← LabelContract JSON 文件，与图片同名
            split.txt ← 每行一个样本 ID

    validate() 检查文件夹结构是否符合约定。
    """
    train_root:      str
    dataset_version: str               = 'combined'
    n_samples:       int               = 0
    label_source:    str               = 'model'  # 'model' | 'manual' | 'mixed'
    min_iou:         float             = 0.0      # 这批标注的最低 IoU
    avg_iou:         float             = 0.0      # 这批标注的平均 IoU
    created_at:      str               = field(default_factory=lambda: __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ'))
    metadata:        Dict[str, Any]    = field(default_factory=dict)

    REQUIRED_SUBDIRS = ('images', 'labels')

    def validate(self):
        if not os.path.exists(self.train_root):
            raise ValidationError(f'train_root 不存在: {self.train_root}')
        for subdir in self.REQUIRED_SUBDIRS:
            path = os.path.join(self.train_root, subdir)
            if not os.path.exists(path):
                raise ValidationError(
                    f'缺少必要子目录: {path}\n'
                    f'期望结构: {self.train_root}/images/, {self.train_root}/labels/'
                )
        if self.n_samples <= 0:
            raise ValidationError(f'n_samples 必须 > 0，当前: {self.n_samples}')

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def mock(cls, train_root: str = '/tmp/mock_train') -> TrainingContract:
        os.makedirs(os.path.join(train_root, 'images'), exist_ok=True)
        os.makedirs(os.path.join(train_root, 'labels'), exist_ok=True)
        return cls(
            train_root      = train_root,
            dataset_version = 'mock',
            n_samples       = 10,
            label_source    = 'mock',
            avg_iou         = 0.5,
        )


# ══════════════════════════════════════════════════════════════
# 契约3：3D 重建输入（ReconstructContract）
# ══════════════════════════════════════════════════════════════

@dataclass
class ReconstructContract:
    """
    3D 重建 Agent 接收的矢量中间件格式。
    由标注契约 + 物理参数组成。

    输出：
        glb_path    → .glb 3D 模型文件路径
    """
    image_path:      str
    image_wh:        Tuple[int, int]
    wall_boxes:      List[Dict]         # BBox.to_list() 的列表，JSON 友好
    openings:        List[Dict]         # Opening.to_dict() 的列表
    pixels_per_meter: float = 50.0
    wall_height_m:   float  = 2.8
    door_height_m:   float  = 2.1
    window_height_m: float  = 1.2
    window_sill_m:   float  = 0.9
    glb_path:        Optional[str] = None

    def validate(self):
        if not self.wall_boxes:
            raise ValidationError('ReconstructContract.wall_boxes 为空，无法重建')
        if self.pixels_per_meter <= 0:
            raise ValidationError(f'pixels_per_meter 必须 > 0: {self.pixels_per_meter}')

    @classmethod
    def from_label(cls, label: LabelContract,
                   pixels_per_meter: float = 50.0) -> ReconstructContract:
        """从 LabelContract 转换，统一入口。"""
        return cls(
            image_path       = label.image_path,
            image_wh         = label.image_wh,
            wall_boxes       = [
                {'bbox': w.bbox.to_list(), 'orientation': w.orientation}
                for w in label.wall_boxes
            ],
            openings         = [
                {'type': o.type, 'bbox': o.bbox.to_list(),
                 'wall_side': o.wall_side,
                 'estimated_width_m': o.estimated_width_m}
                for o in label.openings
            ],
            pixels_per_meter = pixels_per_meter,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def mock(cls, image_path: str = '/tmp/mock.png') -> ReconstructContract:
        return cls(
            image_path = image_path,
            image_wh   = (512, 512),
            wall_boxes = [
                {'bbox': [0, 0, 512, 10],   'orientation': 'horizontal'},
                {'bbox': [0, 502, 512, 512], 'orientation': 'horizontal'},
                {'bbox': [0, 0, 10, 512],   'orientation': 'vertical'},
                {'bbox': [502, 0, 512, 512], 'orientation': 'vertical'},
            ],
            openings = [
                {'type': 'door', 'bbox': [226, 0, 286, 10],
                 'wall_side': 'north', 'estimated_width_m': 0.9},
            ],
        )


# ══════════════════════════════════════════════════════════════
# 契约校验工具
# ══════════════════════════════════════════════════════════════

def validate_handoff(contract, stage_name: str) -> bool:
    """
    Agent 交接时调用，校验契约格式。
    返回 True = 合格，ValidationError = 不合格（调用方决定 retry 或 fail）。
    """
    try:
        contract.validate()
        logger.info(f'[契约校验] {stage_name} ✓')
        return True
    except ValidationError as e:
        logger.error(f'[契约校验] {stage_name} ✗  {e}')
        raise
