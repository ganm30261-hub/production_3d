# check_env.py
# 运行：python check_env.py
# 检查所有依赖是否就绪，输出缺失列表和安装命令

import sys

checks = []

def check(name, import_str, pip_cmd=None):
    try:
        exec(import_str)
        checks.append(("✓", name, ""))
    except ImportError as e:
        checks.append(("✗", name, pip_cmd or f"pip install {name.lower()}"))

# ── 核心依赖 ──
check("torch",               "import torch")
check("torchvision",         "import torchvision")
check("numpy",               "import numpy")
check("cv2",                 "import cv2")
check("PIL",                 "from PIL import Image")
check("albumentations",      "import albumentations")
check("timm",                "import timm")
check("peft",                "from peft import LoraConfig")
check("anthropic",           "import anthropic")
check("mlflow",              "import mlflow")
check("tqdm",                "from tqdm.auto import tqdm")

# ── Agent 依赖 ──
check("faiss",               "import faiss",               "pip install faiss-cpu")
check("sentence_transformers","from sentence_transformers import SentenceTransformer",
                              "pip install sentence-transformers")
check("langgraph",           "from langgraph.graph import StateGraph", "pip install langgraph")

# ── 可选依赖 ──
check("SAM2",                "from sam2.build_sam import build_sam2",
                              "pip install segment-anything-2  # 可选，跳过也能跑")

print("\n── 依赖检查结果 ──\n")
missing = []
for symbol, name, cmd in checks:
    print(f"  {symbol} {name}")
    if symbol == "✗":
        missing.append(cmd)

if missing:
    print("\n── 需要安装 ──\n")
    for cmd in missing:
        print(f"  {cmd}")
    print("\n一键安装（非 SAM2）：")
    core = [c for c in missing if "segment" not in c]
    print("  pip install " + " ".join(
        c.replace("pip install ", "").split()[0] for c in core
    ))
else:
    print("\n  全部就绪 ✓")

# ── CUDA 检查 ──
try:
    import torch
    print(f"\n── CUDA ──")
    print(f"  torch版本: {torch.__version__}")
    print(f"  CUDA可用:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
    else:
        print("  （CPU 模式，mock 跑通没问题，训练需要 GPU）")
except:
    pass

# ── 路径检查 ──
import os
print(f"\n── 路径检查 ──")
paths = {
    "PROJECT_ROOT": os.path.exists('/workspace/production_3d') or os.path.exists('/content'),
    "当前目录":      os.path.exists('./config.py'),
}
for name, ok in paths.items():
    print(f"  {'✓' if ok else '✗'} {name}")
