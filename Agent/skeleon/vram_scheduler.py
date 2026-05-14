# skeleon/vram_scheduler.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

显存调度：集成期核心要素1

RTX 5060 只有一张卡，三个模型串行使用：
    DINOv2+LoRA  →  SAM2  →  训练（独占）

调度规则：
    1. 同一时刻最多一个推理模型在 GPU 上
    2. 每个模型用完后立即卸载（always_unload 策略）
    3. 训练启动前强制卸载所有推理模型 + 获取 GPU 独占锁
    4. OOM 发生时自动清理并重试一次
"""

from __future__ import annotations

import gc
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

from config import logger, DEVICE
from skeleon.tool_configs import VRAM_CFG, DINOV2_CFG, SAM2_CFG, VRAMConfig


# ══════════════════════════════════════════════════════════════
# 显存监控
# ══════════════════════════════════════════════════════════════

def vram_status() -> Dict[str, int]:
    """返回当前显存状态（MB）。无 GPU 时全零。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return {'total': 0, 'used': 0, 'free': 0}
        total = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
        used  = torch.cuda.memory_allocated(0)  // 1024 // 1024
        rsvd  = torch.cuda.memory_reserved(0)   // 1024 // 1024
        return {'total': total, 'used': used, 'reserved': rsvd, 'free': total - rsvd}
    except Exception:
        return {'total': 0, 'used': 0, 'reserved': 0, 'free': 0}


def flush_vram() -> None:
    """释放当前进程的显存碎片。"""
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    mem = vram_status()
    if mem['total'] > 0:
        logger.info(f'[VRAM] 清理后 free={mem["free"]}MB / total={mem["total"]}MB')


def assert_vram_enough(required_mb: int, label: str = '') -> None:
    """
    检查显存是否足够，不足时抛 RuntimeError（触发 HardwareError 分类）。
    required_mb=0 时跳过检查（VLM API 调用不占本地显存）。
    """
    if required_mb == 0:
        return
    mem  = vram_status()
    free = mem.get('free', 0)
    if free == 0:
        return   # CPU 模式，不检查
    if free < required_mb:
        raise RuntimeError(
            f'VRAM 不足 [{label}]: 需要 {required_mb}MB，剩余 {free}MB。'
            f'请先卸载其他模型。'
        )


# ══════════════════════════════════════════════════════════════
# 模型加载器注册表
# ══════════════════════════════════════════════════════════════

def _load_dinov2() -> Any:
    """加载 DINOv2+LoRA 模型到 GPU。"""
    import torch
    from model.model import DINOv2LoRAModel
    from config import CFG

    ckpt_path = DINOV2_CFG.find_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError(
            f'找不到 DINOv2 checkpoint，搜索路径: {DINOV2_CFG.ckpt_search_dirs}'
        )
    model = DINOv2LoRAModel(CFG).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    logger.info(f'[VRAM] DINOv2 加载完成  ckpt={os.path.basename(ckpt_path)}')
    return model


def _load_sam2() -> Any:
    """加载 SAM2 ImagePredictor。"""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not os.path.exists(SAM2_CFG.ckpt_path):
        raise FileNotFoundError(
            f'SAM2 权重不存在: {SAM2_CFG.ckpt_path}\n'
            f'下载: wget -P /workspace/checkpoints '
            f'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt'
        )
    sam2_model = build_sam2(SAM2_CFG.model_cfg, SAM2_CFG.ckpt_path, device=DEVICE)
    predictor  = SAM2ImagePredictor(sam2_model)
    logger.info('[VRAM] SAM2 加载完成')
    return predictor


# 模型名 → (加载函数, 显存预算MB)
MODEL_LOADERS: Dict[str, tuple] = {
    'dinov2': (_load_dinov2, DINOV2_CFG.vram_required_mb),
    'sam2':   (_load_sam2,   SAM2_CFG.vram_required_mb),
}


# ══════════════════════════════════════════════════════════════
# VRAMScheduler：串行模型生命周期管理
# ══════════════════════════════════════════════════════════════

class VRAMScheduler:
    """
    串行显存调度器。

    用法：
        scheduler = VRAMScheduler()

        with scheduler.load('dinov2') as model:
            result = run_inference(image, model)
        # 退出时自动卸载 dinov2

        with scheduler.load('sam2') as predictor:
            refined = refine_mask(image, mask, predictor)
        # 退出时自动卸载 sam2
    """

    def __init__(self, cfg: VRAMConfig = VRAM_CFG):
        self.cfg      = cfg
        self._current: Optional[str] = None
        self._model:   Optional[Any] = None

    @contextmanager
    def load(self, name: str, loader: Callable = None, required_mb: int = None):
        """
        加载模型，yield 给调用方使用，退出时自动卸载。

        name        : 模型名（'dinov2' / 'sam2' / 自定义）
        loader      : 自定义加载函数（None 时用内置 MODEL_LOADERS）
        required_mb : 显存预算（None 时从 MODEL_LOADERS 读取）
        """
        # 确定加载函数和显存预算
        if loader is None:
            if name not in MODEL_LOADERS:
                raise ValueError(f'未知模型: {name}，请传入 loader 参数')
            loader, default_mb = MODEL_LOADERS[name]
            required_mb = required_mb or default_mb
        required_mb = required_mb or 0

        # always_unload 策略：加载新模型前先卸载当前模型
        if self.cfg.unload_strategy == 'always_unload' and self._current:
            self._unload()

        # 显存预检
        flush_vram()
        assert_vram_enough(required_mb, label=name)

        # 加载
        t0 = time.time()
        logger.info(f'[VRAMScheduler] 加载 {name}  需要 {required_mb}MB')
        try:
            model = loader()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                # OOM：清理后重试一次
                logger.warning(f'[VRAMScheduler] OOM，清理后重试: {name}')
                self._unload()
                flush_vram()
                model = loader()
            else:
                raise

        self._current = name
        self._model   = model

        mem = vram_status()
        logger.info(
            f'[VRAMScheduler] {name} 就绪  '
            f'{time.time()-t0:.1f}s  '
            f'GPU used={mem["used"]}MB / total={mem["total"]}MB'
        )

        try:
            yield model
        finally:
            self._unload()

    def _unload(self) -> None:
        """卸载当前模型，释放显存。"""
        if self._model is None:
            return
        name = self._current
        try:
            import torch
            if hasattr(self._model, 'cpu'):
                self._model.cpu()
            del self._model
            self._model   = None
            self._current = None
            flush_vram()
            logger.info(f'[VRAMScheduler] {name} 已卸载')
        except Exception as e:
            logger.warning(f'[VRAMScheduler] 卸载 {name} 失败: {e}')

    def unload_all(self) -> None:
        """强制卸载所有模型（训练前调用）。"""
        self._unload()
        flush_vram()

    def status(self) -> Dict:
        """返回当前调度器状态。"""
        return {
            'current_model': self._current,
            'vram':          vram_status(),
            'strategy':      self.cfg.unload_strategy,
        }


# ══════════════════════════════════════════════════════════════
# 全局单例
# ══════════════════════════════════════════════════════════════

SCHEDULER = VRAMScheduler()
