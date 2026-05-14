# resource_manager.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

资源调度：支柱4 —— 显存管理 + GPU 独占锁

RTX 5060 只有一张卡，是最稀缺的资源。
这个模块负责：
    1. 模型加载/卸载（用完即释放显存）
    2. GPU 独占锁（训练时阻止其他进程占卡）
    3. 显存监控（OOM 前预警）
    4. 训练隔离（训练启动前强制检查显存占用）
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import fcntl
import gc
import os
import subprocess
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

from config import logger, DEVICE

# GPU 独占锁文件路径
_GPU_LOCK_PATH = os.environ.get('GPU_LOCK_FILE', '/tmp/floorplan_gpu.lock')

# 显存阈值（MB），低于此值时发出 OOM 预警
_OOM_WARN_MB = 1024   # 1GB


# ══════════════════════════════════════════════════════════════
# 显存工具
# ══════════════════════════════════════════════════════════════

def gpu_memory_mb() -> Dict[str, int]:
    """
    返回当前 GPU 显存状态（MB）。
    无 GPU 时返回全零。
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {'total': 0, 'used': 0, 'free': 0}
        total = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
        used  = torch.cuda.memory_allocated(0)               // 1024 // 1024
        rsvd  = torch.cuda.memory_reserved(0)                // 1024 // 1024
        free  = total - rsvd
        return {'total': total, 'used': used, 'reserved': rsvd, 'free': free}
    except Exception:
        return {'total': 0, 'used': 0, 'reserved': 0, 'free': 0}


def clear_gpu_memory() -> None:
    """
    释放当前进程占用的显存。
    调用时机：每个 Agent 完成工作后，交接给下一个之前。
    """
    try:
        import torch
        torch.cuda.empty_cache()
        gc.collect()
        mem = gpu_memory_mb()
        logger.info(
            f'[ResourceManager] 显存已释放  '
            f'free={mem["free"]}MB / total={mem["total"]}MB'
        )
    except Exception as e:
        logger.warning(f'[ResourceManager] 清理显存失败: {e}')


def check_oom_risk(required_mb: int = 4000) -> bool:
    """
    检查是否有 OOM 风险。
    required_mb : 本次操作预计需要的显存（MB）
    返回 True = 有风险，应先清理或等待。
    """
    mem  = gpu_memory_mb()
    free = mem.get('free', 0)
    if free == 0:
        return False   # CPU 模式，不检查

    risk = free < required_mb
    if risk:
        logger.warning(
            f'[ResourceManager] OOM 风险: '
            f'需要 {required_mb}MB，剩余 {free}MB'
        )
    return risk


# ══════════════════════════════════════════════════════════════
# GPU 独占锁
# ══════════════════════════════════════════════════════════════

class GPULock:
    """
    基于文件锁的 GPU 独占锁。
    训练启动时获取，训练结束时释放。
    其他 Agent 调用 acquire() 时会等待，不会强抢。

    用法：
        lock = GPULock()
        with lock.acquire(owner="training_agent", timeout=300):
            train_model(...)
        # 退出 with 自动释放
    """

    def __init__(self, lock_path: str = _GPU_LOCK_PATH):
        self.lock_path = lock_path
        self._fd       = None

    @contextmanager
    def acquire(self, owner: str = 'unknown', timeout: int = 600):
        """
        获取 GPU 独占锁，最多等待 timeout 秒。
        获取成功后写入 owner 信息（调试用）。
        """
        t0 = time.time()
        logger.info(f'[GPULock] {owner} 尝试获取 GPU 锁...')

        self._fd = open(self.lock_path, 'w')
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break   # 获取成功
            except BlockingIOError:
                if time.time() - t0 > timeout:
                    self._fd.close()
                    raise TimeoutError(
                        f'[GPULock] 等待 GPU 锁超时 ({timeout}s)  owner={owner}'
                    )
                logger.info(
                    f'[GPULock] GPU 被占用，等待中...  '
                    f'elapsed={int(time.time()-t0)}s'
                )
                time.sleep(10)

        # 写入 owner 信息
        self._fd.write(f'{{"owner": "{owner}", "pid": {os.getpid()}, '
                       f'"acquired_at": "{time.strftime("%Y-%m-%dT%H:%M:%SZ")}"}}\n')
        self._fd.flush()
        logger.info(f'[GPULock] {owner} 获取 GPU 锁成功')

        try:
            yield self
        finally:
            self.release(owner)

    def release(self, owner: str = 'unknown') -> None:
        if self._fd:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                self._fd.close()
                self._fd = None
                logger.info(f'[GPULock] {owner} 释放 GPU 锁')
            except Exception as e:
                logger.warning(f'[GPULock] 释放锁失败: {e}')

    def is_locked(self) -> bool:
        """检查 GPU 是否被其他进程锁定。"""
        if not os.path.exists(self.lock_path):
            return False
        try:
            fd = open(self.lock_path, 'w')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()
            return False
        except BlockingIOError:
            return True

    def who_owns(self) -> Optional[dict]:
        """读取当前锁的持有者信息。"""
        if not os.path.exists(self.lock_path):
            return None
        try:
            import json as _json
            with open(self.lock_path) as f:
                return _json.loads(f.read().strip())
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════
# 模型生命周期管理
# ══════════════════════════════════════════════════════════════

class ModelRegistry:
    """
    管理各 Agent 使用的模型实例，控制加载/卸载时机。

    设计原则：
        - 同一时刻只有一个模型在 GPU 上（训练除外）
        - 每个 Agent 用完后调用 unload()，显存立即释放
        - 训练模式下独占 GPU，不允许推理模型加载

    用法：
        registry = ModelRegistry()

        # 标注 Agent 加载 DINOv2
        with registry.use('dinov2', loader=load_dinov2) as model:
            run_inference(image, model)
        # 退出 with 时自动卸载

        # SAM2（按需加载，用完卸载）
        with registry.use('sam2', loader=load_sam2) as predictor:
            refine_mask(image, mask, predictor)
    """

    def __init__(self):
        self._models: Dict[str, Any] = {}

    @contextmanager
    def use(self, name: str, loader: Callable, required_mb: int = 3000):
        """
        加载模型，在 with 块里使用，退出时卸载。

        name        : 模型标识符（日志用）
        loader      : 无参函数，返回模型实例
        required_mb : 预计需要的显存（用于 OOM 预检）
        """
        # OOM 预检：显存不足时先清理
        if check_oom_risk(required_mb):
            logger.warning(f'[ModelRegistry] 加载 {name} 前先清理显存')
            self.unload_all()
            clear_gpu_memory()

        logger.info(f'[ModelRegistry] 加载模型: {name}')
        t0    = time.time()
        model = loader()
        self._models[name] = model

        mem = gpu_memory_mb()
        logger.info(
            f'[ModelRegistry] {name} 加载完成  '
            f'{time.time()-t0:.1f}s  '
            f'GPU used={mem["used"]}MB'
        )
        try:
            yield model
        finally:
            self.unload(name)

    def unload(self, name: str) -> None:
        """卸载指定模型，释放显存。"""
        if name not in self._models:
            return
        try:
            import torch
            model = self._models.pop(name)
            # 把模型移到 CPU 再删除，确保显存释放
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
            clear_gpu_memory()
            logger.info(f'[ModelRegistry] {name} 已卸载')
        except Exception as e:
            logger.warning(f'[ModelRegistry] 卸载 {name} 失败: {e}')

    def unload_all(self) -> None:
        """卸载所有模型（训练启动前调用）。"""
        for name in list(self._models.keys()):
            self.unload(name)
        logger.info('[ModelRegistry] 所有模型已卸载')


# ══════════════════════════════════════════════════════════════
# 训练隔离
# ══════════════════════════════════════════════════════════════

def prepare_for_training(
    registry:     ModelRegistry,
    gpu_lock:     GPULock,
    required_mb:  int = 8000,
    lock_timeout: int = 600,
) -> contextmanager:
    """
    训练启动前的准备：
        1. 卸载所有推理模型
        2. 清理显存
        3. 检查显存是否足够
        4. 获取 GPU 独占锁

    用法：
        with prepare_for_training(registry, gpu_lock):
            train_one_version('combined')
    """
    @contextmanager
    def _ctx():
        # 卸载推理模型
        registry.unload_all()
        clear_gpu_memory()

        mem = gpu_memory_mb()
        if mem['total'] > 0 and mem['free'] < required_mb:
            raise RuntimeError(
                f'训练所需显存不足: 需要 {required_mb}MB，'
                f'当前剩余 {mem["free"]}MB / 总计 {mem["total"]}MB'
            )

        with gpu_lock.acquire(owner='training_agent', timeout=lock_timeout):
            logger.info(
                f'[ResourceManager] 训练模式启动  '
                f'free={mem["free"]}MB / total={mem["total"]}MB'
            )
            yield

        # 训练结束后再次清理
        clear_gpu_memory()
        logger.info('[ResourceManager] 训练模式结束，显存已释放')

    return _ctx()


# ══════════════════════════════════════════════════════════════
# 全局单例（orchestrator 直接 import 使用）
# ══════════════════════════════════════════════════════════════

GPU_LOCK       = GPULock()
MODEL_REGISTRY = ModelRegistry()
