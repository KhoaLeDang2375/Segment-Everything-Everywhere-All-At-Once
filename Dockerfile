# ======================================================
# Dockerfile - SEEM_v1 RunPod Deployment
# Build: docker build -t seem-v1 .
# Run:   docker run --gpus all -p 7860:7860 seem-v1
# ======================================================

FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Metadata
LABEL maintainer="SEEM_v1 Demo"
LABEL description="SEEM_v1 Gradio Demo with SAM-ViT-L backbone"

WORKDIR /workspace

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git wget curl && \
    rm -rf /var/lib/apt/lists/*

# Clone repository
RUN git clone https://github.com/KhoaLeDang2375/Segment-Everything-Everywhere-All-At-Once.git /workspace/seem
WORKDIR /workspace/seem

# Python dependencies (layered for better cache)
# Layer 1: PyTorch (large, changes rarely)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Layer 2: Core requirements (bỏ deepspeed không cần cho demo)
RUN grep -v "deepspeed" assets/requirements/requirements.txt > /tmp/req.txt && \
    pip install --no-cache-dir -r /tmp/req.txt

# Layer 3: Custom packages
RUN pip install --no-cache-dir \
    git+https://github.com/MaureenZOU/detectron2-xyz.git && \
    pip install --no-cache-dir \
    git+https://github.com/openai/whisper.git && \
    pip install --no-cache-dir \
    git+https://github.com/arogozhnikov/einops.git

# Create checkpoints directory
RUN mkdir -p checkpoints

# Download checkpoints (optional - có thể mount volume thay thế)
# Uncomment nếu muốn bake checkpoints vào image (~3.8GB thêm)
# RUN wget -q -O checkpoints/sam_vit_l_0b3195.pth \
#         https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth && \
#     wget -q -O checkpoints/seem_samvitl_v1.pt \
#         https://huggingface.co/xdecoder/SEEM/resolve/main/seem_samvitl_v1.pt

# Patch config to use local checkpoint path
RUN sed -i "s|PRETRAINED:.*|PRETRAINED: 'checkpoints/sam_vit_l_0b3195.pth'|g" \
    configs/seem/samvitl_unicl_lang_v1.yaml || true

# Copy demo scripts
COPY demo_v1.py /workspace/seem/demo_v1.py
COPY setup.sh /workspace/seem/setup.sh
COPY runpod_start.sh /workspace/seem/runpod_start.sh
RUN chmod +x setup.sh runpod_start.sh

# Expose Gradio port
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:7860 || exit 1

# Default: download checkpoints if missing, then launch
CMD bash -c "\
    if [ ! -f checkpoints/seem_samvitl_v1.pt ]; then \
    echo 'Downloading checkpoints...'; \
    wget -q -O checkpoints/sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth; \
    wget -q -O checkpoints/seem_samvitl_v1.pt https://huggingface.co/xdecoder/SEEM/resolve/main/seem_samvitl_v1.pt; \
    fi && \
    python demo_v1.py --server_name 0.0.0.0 --port 7860"
