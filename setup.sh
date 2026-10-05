#!/bin/bash
# ============================================================
# SEEM_v1 RunPod Setup Script
# Chạy 1 lần sau khi pod khởi động lần đầu
# Usage: bash setup.sh
# ============================================================

set -e
echo "========================================"
echo "  SEEM_v1 RunPod Setup"
echo "========================================"

# 1. Cài system dependencies
echo "[1/6] Installing system packages..."
apt-get update -y -q
apt-get install -y ffmpeg git wget curl -q

# 2. PyTorch — kiểm tra hoặc cài đặt torch compatible (khuyên dùng torch 2.1.2 cu121)
echo "[2/6] Checking PyTorch..."
PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python version: $PYTHON_VER"

if [[ "$PYTHON_VER" == "3.12"* ]] || [[ "$PYTHON_VER" == "3.13"* ]]; then
    echo "  [CẢNH BÁO] Bạn đang dùng Python $PYTHON_VER!"
    echo "  SEEM và Detectron2 yêu cầu Python 3.9 hoặc 3.10."
    echo "  Vui lòng dùng Conda Python 3.10 để tránh lỗi 'pkg_resources' và build C++."
fi

TORCH_OK=$(python3 -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.is_available()}')" 2>/dev/null || echo "NOT_FOUND")
echo "  Found: $TORCH_OK"
if [[ "$TORCH_OK" == "NOT_FOUND" ]]; then
    echo "  PyTorch not found — installing PyTorch 2.1.2 (cu121)..."
    pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 -q
else
    echo "  OK — using existing PyTorch ($TORCH_OK)"
fi

# 3. Đảm bảo setuptools tương thích
echo "[3/6] Setting up setuptools & wheel..."
pip install --no-cache-dir -q "setuptools<70" "wheel" "packaging"

# 4. Cài Python packages
echo "[4/6] Installing Python packages..."
pip install --no-cache-dir -q \
    "pillow<=10.0.1" \
    "opencv-python==4.8.1.78" \
    "pyyaml==6.0.1" \
    "json_tricks==3.17.3" \
    "yacs==0.1.8" \
    "scikit-learn==1.3.1" \
    "pandas==2.0.3" \
    "timm==0.4.12" \
    "numpy>=1.24,<2.0" \
    "einops==0.7.0" \
    "fvcore==0.1.5.post20221221" \
    "transformers==4.34.0" \
    "sentencepiece==0.1.99" \
    "ftfy==6.1.1" \
    "regex==2023.10.3" \
    "nltk==3.8.1" \
    "pycocotools==2.0.7" \
    "shapely" \
    "scikit-image==0.21.0" \
    "accelerate==0.23.0" \
    "kornia==0.7.0" \
    "wandb==0.15.12" \
    "gradio==3.42.0"

# 5. Cài custom packages
echo "[5/6] Installing custom packages (detectron2, whisper)..."
pip install --no-cache-dir --no-build-isolation -q \
    git+https://github.com/MaureenZOU/detectron2-xyz.git

pip install --no-cache-dir -q \
    git+https://github.com/openai/whisper.git

# 6. Tải checkpoints
echo "[6/6] Downloading checkpoints..."
mkdir -p checkpoints

if [ ! -f "checkpoints/sam_vit_l_0b3195.pth" ]; then
    echo "  -> Downloading SAM ViT-L weights (~2.6GB)..."
    wget -q --show-progress \
        -O checkpoints/sam_vit_l_0b3195.pth \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
else
    echo "  -> sam_vit_l_0b3195.pth already exists, skipping."
fi

if [ ! -f "checkpoints/seem_samvitl_v1.pt" ]; then
    echo "  -> Downloading SEEM_v1 SAM-ViT-L checkpoint (~1.2GB)..."
    wget -q --show-progress \
        -O checkpoints/seem_samvitl_v1.pt \
        https://huggingface.co/xdecoder/SEEM/resolve/main/seem_samvitl_v1.pt
else
    echo "  -> seem_samvitl_v1.pt already exists, skipping."
fi

# 7. Patch config YAML - sửa đường dẫn pretrained nếu vẫn còn hardcoded
CONFIG_FILE="configs/seem/samvitl_unicl_lang_v1.yaml"
if grep -q "/nobackup3/" "$CONFIG_FILE" 2>/dev/null; then
    sed -i "s|PRETRAINED:.*|PRETRAINED: 'checkpoints/sam_vit_l_0b3195.pth'|g" "$CONFIG_FILE"
    echo "  -> Patched PRETRAINED path in $CONFIG_FILE"
fi

echo ""
echo "========================================"
echo "  Setup Complete!"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}  GPU OK: {torch.cuda.is_available()}')"
echo "  Run: python demo_v1.py"
echo "  URL: http://0.0.0.0:7860"
echo "========================================"
