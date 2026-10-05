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
apt-get install -y ffmpeg git wget curl

# 2. Cài PyTorch — tự động chọn phiên bản phù hợp với CUDA
echo "[2/6] Installing PyTorch (auto-detect CUDA)..."

# Detect CUDA version
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP "(?<=release )\d+\.\d+" | head -1)
if [ -z "$CUDA_VER" ]; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[\d.]+" | head -1)
fi
echo "  Detected CUDA: ${CUDA_VER:-unknown}"

# Chọn CUDA index URL
if [[ "$CUDA_VER" == 12.4* ]] || [[ "$CUDA_VER" == 12.5* ]] || [[ "$CUDA_VER" == 12.6* ]] || [[ "$CUDA_VER" == 12.7* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
fi

# torch 2.4.1 là phiên bản ổn định mới nhất có trên cả cu121 và cu124
echo "  Using index: $TORCH_INDEX"
pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    --index-url "$TORCH_INDEX" -q

# 3. Cài Python packages từ requirements.txt
echo "[3/6] Installing Python packages..."
# Lọc bỏ: deepspeed (không cần demo), torch/torchvision (đã cài ở bước 2)
grep -vE "^(torch|torchvision|deepspeed)==" assets/requirements/requirements.txt \
    > /tmp/req_demo.txt || true
# Ghi đè numpy để tránh xung đột ABI với torch 2.4.x
echo "numpy>=1.24,<2.0" >> /tmp/req_demo.txt
pip install --no-cache-dir -r /tmp/req_demo.txt -q

# 4. Cài custom packages
echo "[4/6] Installing custom packages (detectron2, whisper, einops)..."
pip install --no-cache-dir \
    git+https://github.com/MaureenZOU/detectron2-xyz.git -q
pip install --no-cache-dir \
    git+https://github.com/openai/whisper.git -q
pip install --no-cache-dir \
    git+https://github.com/arogozhnikov/einops.git -q

# 5. Tải checkpoints
echo "[5/6] Downloading checkpoints..."
mkdir -p checkpoints

# SAM ViT-L backbone weights
if [ ! -f "checkpoints/sam_vit_l_0b3195.pth" ]; then
    echo "  -> Downloading SAM ViT-L weights (~2.6GB)..."
    wget -q --show-progress \
        -O checkpoints/sam_vit_l_0b3195.pth \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
else
    echo "  -> sam_vit_l_0b3195.pth already exists, skipping."
fi

# SEEM_v1 checkpoint (SAM-ViT-L)
if [ ! -f "checkpoints/seem_samvitl_v1.pt" ]; then
    echo "  -> Downloading SEEM_v1 SAM-ViT-L checkpoint (~1.2GB)..."
    wget -q --show-progress \
        -O checkpoints/seem_samvitl_v1.pt \
        https://huggingface.co/xdecoder/SEEM/resolve/main/seem_samvitl_v1.pt
else
    echo "  -> seem_samvitl_v1.pt already exists, skipping."
fi

# 6. Patch config YAML - sửa đường dẫn pretrained
echo "[6/6] Patching config files..."
CONFIG_FILE="configs/seem/samvitl_unicl_lang_v1.yaml"
if grep -q "/nobackup3/" "$CONFIG_FILE"; then
    sed -i "s|PRETRAINED:.*|PRETRAINED: 'checkpoints/sam_vit_l_0b3195.pth'|g" "$CONFIG_FILE"
    echo "  -> Patched PRETRAINED path in $CONFIG_FILE"
fi

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "  Run: python demo_v1.py"
echo "  Access: http://0.0.0.0:7860"
echo "========================================"
