#!/bin/bash
# ============================================================
# runpod_start.sh
# Quick start script for SEEM_v1 on RunPod
#
# Usage:
#   bash runpod_start.sh              # SAM-ViT-L (default)
#   bash runpod_start.sh --vitb       # SAM-ViT-B (nhẹ hơn)
#   bash runpod_start.sh --focall     # Focal-L
# ============================================================

set -e

REPO_DIR="/workspace/Segment-Everything-Everywhere-All-At-Once"
CONF_SAMVITL="configs/seem/samvitl_unicl_lang_v1.yaml"
CONF_SAMVITB="configs/seem/samvitb_unicl_lang_v1.yaml"
CONF_FOCALL="configs/seem/focall_unicl_lang_v1.yaml"
CKPT_SAMVITL="checkpoints/seem_samvitl_v1.pt"
CKPT_SAMVITB="checkpoints/seem_samvitb_v1.pt"
CKPT_FOCALL="checkpoints/seem_focall_v1.pt"

# Parse args
CONF=$CONF_SAMVITL
CKPT=$CKPT_SAMVITL
SAM_CKPT="checkpoints/sam_vit_l_0b3195.pth"

for arg in "$@"; do
    case $arg in
        --vitb)
            CONF=$CONF_SAMVITB
            CKPT=$CKPT_SAMVITB
            SAM_CKPT="checkpoints/sam_vit_b_01ec64.pth"
            echo "[INFO] Using SAM-ViT-B backbone"
            ;;
        --focall)
            CONF=$CONF_FOCALL
            CKPT=$CKPT_FOCALL
            SAM_CKPT=""
            echo "[INFO] Using Focal-L backbone"
            ;;
    esac
done

# Check nếu đã clone repo chưa
if [ ! -d "$REPO_DIR" ]; then
    echo "[INFO] Cloning SEEM repo..."
    cd /workspace
    git clone https://github.com/KhoaLeDang2375/Segment-Everything-Everywhere-All-At-Once.git
fi

cd "$REPO_DIR"

# Check nếu cần setup
if [ ! -f "checkpoints/seem_samvitl_v1.pt" ] && [ ! -f "checkpoints/seem_samvitb_v1.pt" ]; then
    echo "[INFO] Running first-time setup..."
    bash setup.sh
fi

# Launch demo
echo "[INFO] Starting SEEM_v1 Demo..."
echo "[INFO] Config: $CONF"
echo "[INFO] Checkpoint: $CKPT"
echo ""

python demo_v1.py \
    --conf_files "$CONF" \
    --checkpoint "$CKPT" \
    --sam_checkpoint "$SAM_CKPT" \
    --port 7860 \
    --server_name 0.0.0.0
