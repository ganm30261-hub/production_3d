# model_promoter.py
"""
支柱4：模型晋升机制

冠军/挑战者模式：
    Champion   当前生产环境使用的模型
    Challenger 新训练出来的模型，需要在验证集上击败 Champion 才能晋升

晋升条件：
    挑战者 val_iou 超过冠军 0.02（防止噪声波动）

回滚条件：
    3D 重建失败率 > 20% 时自动回退到上一个稳定版本
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import logger, PseudoLabelConfig, CFG, DEVICE


# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════

@dataclass
class PromoterConfig:
    # 晋升条件
    min_improvement:    float = 0.02   # 挑战者比冠军高 0.02 才晋升
    min_val_iou:        float = 0.75   # 绝对下限（低于此值无论如何不晋升）

    # 回滚条件
    max_reconstruct_fail_rate: float = 0.20  # 3D 重建失败率上限
    rollback_window:            int   = 20   # 观察最近 N 次重建

    # 文件路径
    champion_registry: str = '/workspace/production_3d/outputs/champion.json'
    ckpt_archive_dir:  str = '/workspace/production_3d/outputs/model_archive'

    # 验证集
    val_gt_dir: str = '/workspace/production_3d/data/val_gt'   # 200 张人工标注真值


DEFAULT_PROMOTER_CFG = PromoterConfig()


# ══════════════════════════════════════════════════════════════
# 冠军注册表
# ══════════════════════════════════════════════════════════════

@dataclass
class ModelRecord:
    """一个模型版本的完整记录。"""
    version_id:    str
    ckpt_path:     str
    val_iou:       float
    promoted_at:   str
    dataset_version: str
    lora_r:        int
    is_champion:   bool   = False
    reconstruct_fail_count: int = 0
    reconstruct_total:      int = 0
    metadata:      dict   = field(default_factory=dict)

    @property
    def reconstruct_fail_rate(self) -> float:
        if self.reconstruct_total == 0:
            return 0.0
        return self.reconstruct_fail_count / self.reconstruct_total

    def to_dict(self) -> dict:
        return {
            'version_id':    self.version_id,
            'ckpt_path':     self.ckpt_path,
            'val_iou':       self.val_iou,
            'promoted_at':   self.promoted_at,
            'dataset_version': self.dataset_version,
            'lora_r':        self.lora_r,
            'is_champion':   self.is_champion,
            'reconstruct_fail_count': self.reconstruct_fail_count,
            'reconstruct_total':      self.reconstruct_total,
            'metadata':      self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelRecord:
        return cls(**d)


class ChampionRegistry:
    """冠军模型注册表，持久化到 JSON 文件。"""

    def __init__(self, path: str = DEFAULT_PROMOTER_CFG.champion_registry):
        self.path    = path
        self.records: List[ModelRecord] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
            self.records = [ModelRecord.from_dict(r) for r in data.get('records', [])]

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(
                {'records': [r.to_dict() for r in self.records],
                 'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ')},
                f, indent=2, ensure_ascii=False,
            )

    @property
    def champion(self) -> Optional[ModelRecord]:
        """当前冠军（最新的 is_champion=True 记录）。"""
        champions = [r for r in self.records if r.is_champion]
        return champions[-1] if champions else None

    def add(self, record: ModelRecord) -> None:
        self.records.append(record)
        self.save()

    def set_champion(self, version_id: str) -> None:
        """把指定版本设为冠军，其他全部降级。"""
        for r in self.records:
            r.is_champion = (r.version_id == version_id)
        self.save()

    def update_reconstruct_stats(
        self, version_id: str, failed: bool
    ) -> None:
        """每次 3D 重建后更新统计。"""
        for r in self.records:
            if r.version_id == version_id:
                r.reconstruct_total      += 1
                r.reconstruct_fail_count += int(failed)
                break
        self.save()


# ══════════════════════════════════════════════════════════════
# 验证集评估
# ══════════════════════════════════════════════════════════════

def evaluate_on_val_set(
    ckpt_path: str,
    cfg:       PseudoLabelConfig,
    val_gt_dir: str,
) -> dict:
    """
    在 200 张人工标注的真值上评估模型。
    返回 {'val_iou': float, 'n_samples': int}
    """
    import torch
    import cv2
    import numpy as np
    from model.model import DINOv2LoRAModel
    from pipeline import run_inference

    if not os.path.exists(val_gt_dir):
        logger.warning(f'[ModelPromoter] 验证集目录不存在: {val_gt_dir}，跳过评估')
        return {'val_iou': 0.0, 'n_samples': 0}

    # 加载模型
    model = DINOv2LoRAModel(cfg).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    iou_list = []
    for img_path in glob.glob(os.path.join(val_gt_dir, '*/F1_scaled.png'))[:200]:
        gt_path = img_path.replace('F1_scaled.png', 'wall_mask_gt.png')
        if not os.path.exists(gt_path):
            continue

        img_rgb  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        gt_mask  = (cv2.imread(gt_path, 0) > 127).astype(np.uint8)
        out      = run_inference(img_rgb, model, cfg)
        pred     = out['wall_mask'].astype(bool)
        gt_bool  = gt_mask.astype(bool)

        inter = (pred & gt_bool).sum()
        union = (pred | gt_bool).sum()
        if union > 0:
            iou_list.append(inter / union)

    avg_iou = float(np.mean(iou_list)) if iou_list else 0.0
    logger.info(f'[ModelPromoter] 验证集评估完成  val_iou={avg_iou:.4f}  n={len(iou_list)}')
    return {'val_iou': avg_iou, 'n_samples': len(iou_list)}


# ══════════════════════════════════════════════════════════════
# ModelPromoter
# ══════════════════════════════════════════════════════════════

class ModelPromoter:
    """
    冠军/挑战者模型晋升逻辑。

    用法：
        promoter = ModelPromoter()

        # 训练完成后，尝试晋升
        result = promoter.try_promote(
            challenger_ckpt = '/path/to/new_best.pth',
            challenger_iou  = 0.82,
            cfg             = CFG,
        )

        # 每次 3D 重建后记录结果
        promoter.record_reconstruct(failed=False)

        # 自动检查是否需要回滚
        promoter.check_rollback()
    """

    def __init__(
        self,
        cfg:      PseudoLabelConfig = CFG,
        promoter_cfg: PromoterConfig = DEFAULT_PROMOTER_CFG,
    ):
        self.cfg          = cfg
        self.promoter_cfg = promoter_cfg
        self.registry     = ChampionRegistry(promoter_cfg.champion_registry)

    def try_promote(
        self,
        challenger_ckpt: str,
        challenger_iou:  float,
        train_cfg:       PseudoLabelConfig = None,
        skip_val_eval:   bool = False,
    ) -> dict:
        """
        尝试让挑战者晋升为冠军。

        skip_val_eval=True 时直接用训练时的 val_iou，
        False 时在独立验证集上重新评估（更准确）。
        """
        train_cfg = train_cfg or self.cfg
        champion  = self.registry.champion

        # 在验证集上评估（更准确）
        if not skip_val_eval and os.path.exists(self.promoter_cfg.val_gt_dir):
            logger.info('[ModelPromoter] 在验证集上评估挑战者...')
            eval_result     = evaluate_on_val_set(
                challenger_ckpt, train_cfg, self.promoter_cfg.val_gt_dir
            )
            challenger_iou  = eval_result['val_iou']

        # 绝对下限检查
        if challenger_iou < self.promoter_cfg.min_val_iou:
            logger.info(
                f'[ModelPromoter] ✗ 挑战者未达绝对下限  '
                f'iou={challenger_iou:.4f} < {self.promoter_cfg.min_val_iou}'
            )
            return {'promoted': False, 'reason': 'below_min_val_iou'}

        # 与冠军比较
        champion_iou = champion.val_iou if champion else 0.0
        gap          = challenger_iou - champion_iou

        if gap < self.promoter_cfg.min_improvement:
            logger.info(
                f'[ModelPromoter] ✗ 挑战者提升不足  '
                f'gap={gap:.4f} < {self.promoter_cfg.min_improvement}'
            )
            return {'promoted': False, 'reason': f'improvement_too_small: {gap:.4f}'}

        # 晋升
        version_id = f'v{time.strftime("%Y%m%d_%H%M%S")}_iou{challenger_iou:.4f}'

        # 归档挑战者权重
        archive_path = self._archive_ckpt(challenger_ckpt, version_id)

        challenger_record = ModelRecord(
            version_id      = version_id,
            ckpt_path       = archive_path,
            val_iou         = challenger_iou,
            promoted_at     = time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            dataset_version = train_cfg.dataset_version,
            lora_r          = train_cfg.lora_r,
            is_champion     = True,
        )

        # 旧冠军降级
        if champion:
            champion.is_champion = False
            self.registry.save()

        self.registry.add(challenger_record)
        self.registry.set_champion(version_id)

        logger.info(
            f'[ModelPromoter] ✓ 挑战者晋升为冠军  '
            f'iou: {champion_iou:.4f} → {challenger_iou:.4f}  '
            f'gap={gap:.4f}  version={version_id}'
        )

        return {
            'promoted':    True,
            'version_id':  version_id,
            'old_iou':     champion_iou,
            'new_iou':     challenger_iou,
            'gap':         gap,
            'ckpt_path':   archive_path,
        }

    def record_reconstruct(self, failed: bool) -> None:
        """每次 3D 重建后调用，记录成功/失败。"""
        champion = self.registry.champion
        if champion:
            self.registry.update_reconstruct_stats(champion.version_id, failed)

    def check_rollback(self) -> bool:
        """
        检查是否需要回滚。
        失败率超标时自动回退到上一个稳定版本。
        返回 True = 触发了回滚。
        """
        champion = self.registry.champion
        if not champion:
            return False

        if champion.reconstruct_total < self.promoter_cfg.rollback_window:
            return False   # 样本不够，不判断

        fail_rate = champion.reconstruct_fail_rate
        if fail_rate <= self.promoter_cfg.max_reconstruct_fail_rate:
            return False

        logger.warning(
            f'[ModelPromoter] 重建失败率过高: {fail_rate:.1%} > '
            f'{self.promoter_cfg.max_reconstruct_fail_rate:.0%}，触发回滚'
        )

        # 找上一个稳定版本
        prev_stable = self._find_prev_stable(champion.version_id)
        if prev_stable is None:
            logger.error('[ModelPromoter] 找不到上一个稳定版本，无法回滚')
            return False

        champion.is_champion     = False
        prev_stable.is_champion  = True
        self.registry.save()

        logger.info(
            f'[ModelPromoter] 回滚完成  '
            f'{champion.version_id} → {prev_stable.version_id}  '
            f'iou={prev_stable.val_iou:.4f}'
        )
        return True

    def get_champion_ckpt(self) -> Optional[str]:
        """获取当前冠军的 checkpoint 路径，推理时调用。"""
        champion = self.registry.champion
        if champion and os.path.exists(champion.ckpt_path):
            return champion.ckpt_path
        # 没有冠军时，找最新的 best.pth
        return self._find_latest_ckpt()

    # ── 内部工具 ──

    def _archive_ckpt(self, src: str, version_id: str) -> str:
        """把 checkpoint 归档到统一目录。"""
        os.makedirs(self.promoter_cfg.ckpt_archive_dir, exist_ok=True)
        dst = os.path.join(
            self.promoter_cfg.ckpt_archive_dir,
            f'{version_id}.pth'
        )
        shutil.copy2(src, dst)
        return dst

    def _find_prev_stable(self, current_version: str) -> Optional[ModelRecord]:
        """找上一个重建失败率低的版本。"""
        for r in reversed(self.registry.records):
            if r.version_id == current_version:
                continue
            if r.reconstruct_fail_rate <= self.promoter_cfg.max_reconstruct_fail_rate:
                return r
        return None

    def _find_latest_ckpt(self) -> Optional[str]:
        """兜底：从 checkpoints 目录找最新的 best.pth。"""
        import glob
        for version in ('combined', 'hq', 'hq_arch'):
            d = f'/workspace/production_3d/checkpoints_dinov2_lora/{version}'
            matches = sorted(glob.glob(os.path.join(d, '*_best.pth')))
            if matches:
                return matches[-1]
        return None


# ── 全局单例 ──
PROMOTER = ModelPromoter()
