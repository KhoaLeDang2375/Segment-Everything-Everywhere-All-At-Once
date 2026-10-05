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

# 2. Cài PyTorch (đảm bảo đúng phiên bản)
echo "[2/6] Installing PyTorch 2.1.0 (CUDA 12.1)..."
pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121 -q

# 3. Cài Python packages từ requirements.txt (bỏ deepspeed nếu lỗi)
echo "[3/6] Installing Python packages..."
# Cài tất cả trừ deepspeed (không cần cho demo)
grep -v "deepspeed" assets/requirements/requirements.txt > /tmp/req_nodeeospeed.txt || true
pip install --no-cache-dir -r /tmp/req_nodeeospeed.txt -q

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
