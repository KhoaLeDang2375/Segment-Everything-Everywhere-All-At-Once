#!/bin/bash
# ============================================================
# setup_conda.sh - Cài đặt Miniconda Python 3.10 trên RunPod
# Dành cho container đang chạy Python 3.12 (tránh dependency hell)
# Usage: bash setup_conda.sh
# ============================================================

set -e
echo "=================================================="
echo "  Setting up Conda (Python 3.10) for SEEM_v1"
echo "=================================================="

# 1. Cài Miniconda nếu chưa có
if [ ! -d "/root/miniconda3" ]; then
    echo "[1/3] Downloading & Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /root/miniconda3
    rm -f /tmp/miniconda.sh
else
    echo "[1/3] Miniconda already installed at /root/miniconda3"
fi

# 2. Khởi tạo conda cho bash
export PATH="/root/miniconda3/bin:$PATH"
eval "$(/root/miniconda3/bin/conda shell.bash hook)"

# 3. Chấp nhận ToS và tạo môi trường seem với Python 3.10 (dùng conda-forge)
echo "[2/3] Creating Conda env 'seem' with Python 3.10..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if conda env list | grep -q "seem"; then
    echo "  Env 'seem' already exists."
else
    conda create -n seem -c conda-forge python=3.10 -y -q
fi

# 4. Kích hoạt và chạy setup.sh
echo "[3/3] Activating 'seem' env and running setup.sh..."
conda activate seem

# Chạy setup.sh bên trong env seem
bash setup.sh

echo ""
echo "=================================================="
echo "  HOÀN TẤT!"
echo "  Mỗi khi mở terminal mới, hãy chạy:"
echo "    source /root/miniconda3/bin/activate seem"
echo "    python demo_v1.py"
echo "=================================================="
