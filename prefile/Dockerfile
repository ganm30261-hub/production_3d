# Dockerfile — GPU 训练镜像（Vast.ai / GCE GPU 实例）
# 基于官方 PyTorch GPU 镜像，避免手动配置 CUDA

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# ── 系统依赖 ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

# ── Python 依赖 ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── CubiCasa5k 解析库 ──
RUN git clone https://github.com/CubiCasa/CubiCasa5k.git /workspace/CubiCasa5k && \
    pip install --no-cache-dir -e /workspace/CubiCasa5k

# ── 项目代码 ──
COPY config.py utils.py model_arch.py train_engine.py predict.py preprocess_runner.py ./

# 日志目录
RUN mkdir -p /workspace/checkpoints_paper /workspace/mlruns /tmp/floorplan_cache

ENV PYTHONUNBUFFERED=1

# 默认启动训练
CMD ["python", "train_engine.py"]
