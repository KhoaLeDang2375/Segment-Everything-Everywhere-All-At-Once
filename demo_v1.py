# --------------------------------------------------------
# SEEM_v1 Demo Entry Point - RunPod Deployment
# Hỗ trợ: SAM-ViT-L và SAM-ViT-B backbone (SEEM_v1)
#
# Usage:
#   # SAM-ViT-L (mạnh nhất, cần ≥16GB VRAM)
#   python demo_v1.py
#
#   # SAM-ViT-B (nhẹ hơn, cần ≥8GB VRAM)
#   python demo_v1.py \
#       --conf_files configs/seem/samvitb_unicl_lang_v1.yaml \
#       --checkpoint checkpoints/seem_samvitb_v1.pt
#
# Truy cập: http://<pod-ip>:7860
# --------------------------------------------------------

import os
import warnings
import argparse
import numpy as np
import torch
import whisper
import gradio as gr
from PIL import Image

warnings.filterwarnings("ignore")

# ── Argument parsing ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser('SEEM_v1 Demo', add_help=True)
    parser.add_argument(
        '--conf_files',
        default="configs/seem/samvitl_unicl_lang_v1.yaml",
        metavar="FILE",
        help='Path to config file (default: SAM-ViT-L)'
    )
    parser.add_argument(
        '--checkpoint',
        default="checkpoints/seem_samvitl_v1.pt",
        help='Path to SEEM_v1 checkpoint'
    )
    parser.add_argument(
        '--sam_checkpoint',
        default="checkpoints/sam_vit_l_0b3195.pth",
        help='Path to SAM backbone weights'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Gradio server port'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        default=False,
        help='Create public Gradio share URL'
    )
    parser.add_argument(
        '--server_name',
        default="0.0.0.0",
        help='Gradio server host (0.0.0.0 for RunPod)'
    )
    return parser.parse_args()


def validate_checkpoints(args):
    """Kiểm tra các file checkpoint có tồn tại không."""
    missing = []
    if not os.path.exists(args.checkpoint):
        missing.append(f"SEEM checkpoint: {args.checkpoint}")
    if not os.path.exists(args.sam_checkpoint):
        missing.append(f"SAM backbone: {args.sam_checkpoint}")
    if missing:
        print("\n[ERROR] Missing checkpoints:")
        for m in missing:
            print(f"  - {m}")
        print("\nChạy: bash setup.sh để tải về tự động")
        exit(1)


def load_model(args):
    """Build và load SEEM_v1 model."""
    from modeling.BaseModel import BaseModel
    from modeling import build_model
    from utils.distributed import init_distributed
    from utils.arguments import load_opt_from_config_files

    print(f"[INFO] Loading config: {args.conf_files}")
    opt = load_opt_from_config_files([args.conf_files])
    opt = init_distributed(opt)

    # Override backbone pretrained path từ args (thay vì hardcoded trong YAML)
    opt['MODEL']['BACKBONE']['PRETRAINED'] = args.sam_checkpoint

    print(f"[INFO] Building SEEM_v1 model...")
    print(f"       Backbone: {opt['MODEL']['BACKBONE']['NAME'].upper()} "
          f"({opt['MODEL']['BACKBONE'].get('VIT', {}).get('SIZE', 'default')})")
    print(f"       Decoder:  {opt['MODEL']['DECODER']['NAME']}")

    model = BaseModel(opt, build_model(opt)).from_pretrained(args.checkpoint).eval().cuda()

    from utils.constants import COCO_PANOPTIC_CLASSES
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            COCO_PANOPTIC_CLASSES + ["background"], is_eval=True
        )

    print("[INFO] Model loaded successfully!")
    return model, opt


def build_gradio_ui(model, audio_model):
    """Xây dựng Gradio interface."""
    from demo.seem.tasks import interactive_infer_image, interactive_infer_video

    @torch.no_grad()
    def inference(image, tasks, ref_image, ref_text, audio_path, video_path):
        """Main inference handler."""
        if not tasks:
            tasks = ["Panoptic"]

        # ImageMask trả về dict {'image': ndarray, 'mask': ndarray}
        # khi tool=sketch. Cần convert sang PIL để model xử lý.
        def to_pil_dict(x):
            """Giữ nguyên dict format mà demo.seem.tasks kỳ vọng."""
            if x is None:
                return None
            if isinstance(x, dict):
                img = x.get('image')   # ndarray HxWx3
                mask = x.get('mask')  # ndarray HxWx4 (RGBA scribble)
                if img is None:
                    return None
                result = {'image': Image.fromarray(img)}
                if mask is not None:
                    result['mask'] = Image.fromarray(mask)
                return result
            # fallback: ảnh thuần (ndarray hoặc PIL)
            if isinstance(x, np.ndarray):
                return {'image': Image.fromarray(x), 'mask': None}
            return {'image': x, 'mask': None}

        image_dict = to_pil_dict(image)
        ref_image_dict = to_pil_dict(ref_image)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if 'Video' in tasks:
                return interactive_infer_video(
                    model, audio_model, image_dict, tasks,
                    ref_image_dict, ref_text, audio_path, video_path
                )
            else:
                return interactive_infer_image(
                    model, audio_model, image_dict, tasks,
                    ref_image_dict, ref_text, audio_path, video_path
                )

    # Sử dụng hàm trả về gr.Image thay vì kế thừa class
    # để tránh lỗi frontend Gradio không nhận dạng được tool="sketch"
    def ImageMask(**kwargs):
        kwargs.pop('type', None)
        return gr.Image(source="upload", tool="sketch", type="numpy", interactive=True, **kwargs)

    title = "SEEM_v1 — Segment Everything Everywhere All at Once"
    description = """
<div style="text-align:center; padding: 12px">
  <h3>SEEM_v1 · SAM-ViT-L Backbone · Multi-object Interactive Segmentation</h3>
  <div>
    [<a href="https://arxiv.org/pdf/2304.06718.pdf" target="_blank">📄 Paper</a>]
    [<a href="https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once" target="_blank">💻 GitHub</a>]
  </div>
  <br>
  <b>Hướng dẫn:</b>
  Upload ảnh → chọn loại prompt → nhập prompt → Submit
</div>
"""

    inputs = [
        ImageMask(label="🖼️ Upload ảnh → Vẽ Scribble/Stroke lên vùng cần segment"),
        gr.CheckboxGroup(
            choices=["Stroke", "Example", "Text", "Audio", "Video", "Panoptic"],
            value=["Panoptic"],
            label="🎛️ Chế độ Interactive"
        ),
        ImageMask(label="🔗 [Example] Referring Image — vẽ lên vùng tham chiếu"),
        gr.Textbox(
            label="📝 [Text] Mô tả đối tượng cần segment",
            placeholder="VD: 'the dog', 'red car', 'person on the left'..."
        ),
        gr.Audio(
            label="🎙️ [Audio] Ghi âm lệnh",
            source="microphone",
            type="filepath"
        ),
        gr.Video(
            label="🎬 [Video] Video để segment",
            format="mp4",
            interactive=True
        ),
    ]

    outputs = [
        gr.Image(type="pil", label="✅ Kết quả Segmentation (Image)"),
        gr.Video(label="✅ Kết quả Segmentation (Video)", format="mp4"),
    ]

    examples = [
        ["demo/seem/examples/corgi1.webp", ["Text"],
         "demo/seem/examples/corgi2.jpg", "The corgi.", None, None],
        ["demo/seem/examples/zebras1.jpg", ["Example"],
         "demo/seem/examples/zebras2.jpg", "", None, None],
        ["demo/seem/examples/fries1.png", ["Example"],
         "demo/seem/examples/fries2.png", "", None, None],
    ]

    return gr.Interface(
        fn=inference,
        inputs=inputs,
        outputs=outputs,
        title=title,
        description=description,
        examples=examples,
        allow_flagging='never',
        cache_examples=False,
    )


def main():
    args = parse_args()

    print("\n" + "="*50)
    print("  SEEM_v1 Demo - RunPod")
    print("="*50)
    print(f"  Config     : {args.conf_files}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  SAM weights: {args.sam_checkpoint}")
    print(f"  Port       : {args.port}")
    print("="*50 + "\n")

    # Validate
    validate_checkpoints(args)

    # Load model
    model, opt = load_model(args)

    # Load Whisper for audio
    print("[INFO] Loading Whisper (audio model)...")
    audio_model = whisper.load_model("base")
    print("[INFO] Whisper loaded!")

    # Build and launch UI
    demo = build_gradio_ui(model, audio_model)

    print(f"\n[INFO] Launching Gradio on http://{args.server_name}:{args.port}")
    print(f"[INFO] RunPod URL: https://<pod-id>-{args.port}.proxy.runpod.net\n")

    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
