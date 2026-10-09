#!/usr/bin/env python3
"""RunPod Gradio entry point, sharing SEEM's tested demo_v1 loader."""
import argparse
import os
from pathlib import Path
from relocation.config import REPO, Settings


def parse_args():
    parser = argparse.ArgumentParser(description='SEEM v1 + BrushNet relocation')
    defaults = Settings()
    for name in ['seem_config', 'seem_checkpoint', 'sam_checkpoint', 'brushnet_repo',
                 'brushnet_python', 'depth_python', 'lama_python', 'lama_checkpoint', 'base_model', 'brushnet_checkpoint',
                 'removal_brushnet_checkpoint', 'ip_adapter_dir', 'output_dir']:
        parser.add_argument('--' + name.replace('_', '-'), default=getattr(defaults, name))
    parser.add_argument('--max-side', type=int, choices=[512, 768, 1024], default=512)
    parser.add_argument('--worker-timeout', type=int, default=1800)
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--server-name', default='0.0.0.0')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--preflight', action='store_true', help='Check environments and checkpoint paths without loading models')
    return parser.parse_args()


def preflight(settings):
    import subprocess
    import sys
    failures = []
    for label, path in [('SEEM checkpoint', settings.seem_checkpoint), ('SAM weights', settings.sam_checkpoint),
                        ('LaMa checkpoint', settings.lama_checkpoint),
                        ('Base model', settings.base_model), ('BrushNet checkpoint', settings.brushnet_checkpoint),
                        ('Removal BrushNet checkpoint', Path(settings.removal_brushnet_checkpoint) / 'diffusion_pytorch_model.safetensors'),
                        ('IP-Adapter Plus', Path(settings.ip_adapter_dir) / 'models/ip-adapter-plus_sd15.safetensors'),
                        ('IP-Adapter encoder', Path(settings.ip_adapter_dir) / 'models/image_encoder/model.safetensors'),
                        ('BrushNet source', Path(settings.brushnet_repo) / 'src/diffusers')]:
        if not Path(path).exists():
            failures.append(f'{label}: missing {path}')
    checks = [
        ('SEEM', sys.executable, "import torch,gradio,detectron2; assert torch.cuda.is_available(); print('torch',torch.__version__,'gradio',gradio.__version__); print(torch.cuda.get_device_name())"),
        ('BrushNet', settings.brushnet_python, "import diffusers,torch; from diffusers import BrushNetModel,StableDiffusionBrushNetPipeline; from transformers import CLIPVisionModelWithProjection; assert hasattr(StableDiffusionBrushNetPipeline,'load_ip_adapter'); print(diffusers.__file__); assert torch.cuda.is_available()"),
        ('Depth', settings.depth_python, "import torch,transformers; from transformers import AutoModelForDepthEstimation; print(transformers.__version__); assert torch.cuda.is_available()"),
        ('LaMa', settings.lama_python, "import torch,numpy,PIL; assert torch.cuda.is_available(); print(torch.__version__)"),
    ]
    for label, python, code in checks:
        if not Path(python).is_file():
            failures.append(f'{label}: missing Python {python}')
            continue
        completed = subprocess.run([str(python), '-c', code], capture_output=True, text=True)
        print(f'[{label}] {completed.stdout.strip()}')
        if completed.returncode:
            failures.append(f'{label}: {completed.stderr.strip()}')
    if failures:
        raise SystemExit('\n'.join(failures))
    print('Preflight OK. This confirms imports/paths, not model inference.')


def main():
    args = parse_args()
    os.chdir(REPO)
    settings = Settings(**{name: getattr(args, name) for name in Settings.__dataclass_fields__})
    if args.preflight:
        preflight(settings)
        return
    from relocation.ui import build_ui
    ui = build_ui(settings)
    ui.queue(concurrency_count=1, max_size=8).launch(server_name=args.server_name,
        server_port=args.port, share=args.share, show_error=True)


if __name__ == '__main__':
    main()
