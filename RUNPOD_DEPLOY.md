# SEEM_v1 RunPod Demo — Hướng dẫn Triển khai

## Tổng quan

Demo **SEEM_v1** (SAM-ViT-L backbone) chạy trên RunPod với Gradio Web UI.  
Hỗ trợ: **Stroke · Text · Audio · Referring Image · Video · Panoptic segmentation**

## Cấu trúc Files đã tạo

```
├── demo_v1.py          ← Entry point chính cho SEEM_v1 (thay thế demo/seem/app.py)
├── setup.sh            ← Cài đặt môi trường lần đầu
├── runpod_start.sh     ← Script khởi động nhanh
└── Dockerfile          ← Container hoá (tuỳ chọn)
```

## Bước 1: Tạo RunPod Instance

1. Truy cập [runpod.io](https://www.runpod.io/)
2. **GPU**: RTX A4000 (24GB) hoặc RTX 3090 (24GB) — chi phí ~$0.44/hr
3. **Template**: `RunPod PyTorch 2.1.0` (CUDA 12.1)
4. **Container Disk**: 50 GB
5. **Expose HTTP Port**: `7860`

## Bước 2: Cài đặt (chạy trong Terminal pod)

```bash
# Clone repo (nếu chưa có)
git clone https://github.com/KhoaLeDang2375/Segment-Everything-Everywhere-All-At-Once.git
cd Segment-Everything-Everywhere-All-At-Once

# Chạy setup (cài deps + tải checkpoints ~3.8GB)
bash setup.sh
```

## Bước 3: Chạy Demo

```bash
# SAM-ViT-L (mặc định, tốt nhất)
python demo_v1.py

# Hoặc dùng script nhanh
bash runpod_start.sh

# SAM-ViT-B (nếu VRAM < 16GB)
bash runpod_start.sh --vitb
```

## Bước 4: Truy cập Demo

- **Trực tiếp qua RunPod**: `https://<pod-id>-7860.proxy.runpod.net`
- **SSH tunnel**: `ssh -L 7860:localhost:7860 root@<pod-ip>`

## Checkpoints Hỗ trợ

| Model | Backbone | VRAM | Performance |
|-------|---------|------|------------|
| `seem_samvitl_v1.pt` | SAM-ViT-L | ≥16GB | NoC85: 2.40 ✨ |
| `seem_samvitb_v1.pt` | SAM-ViT-B | ≥8GB  | NoC85: 2.53 |
| `seem_focall_v1.pt`  | Focal-L   | ≥16GB | NoC85: 2.66 |
| `seem_focalt_v1.pt`  | Focal-T   | ≥8GB  | NoC85: 3.19 |

## Lỗi thường gặp

| Lỗi | Giải pháp |
|-----|---------|
| `ModuleNotFoundError: detectron2` | `pip install git+https://github.com/MaureenZOU/detectron2-xyz.git` |
| `CUDA out of memory` | Dùng `--vitb` hoặc tăng GPU |
| `FileNotFoundError: sam_vit_l...` | Chạy lại `bash setup.sh` |
| Port 7860 không truy cập được | Thêm HTTP port 7860 trong RunPod pod settings |
| `PRETRAINED path not found` | `setup.sh` tự patch, hoặc sửa thủ công trong YAML |
