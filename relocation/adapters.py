"""Reuse demo_v1's model loader; isolate BrushNet/depth in short-lived workers."""
from argparse import Namespace
from pathlib import Path
import gc
import json
import os
import subprocess
import numpy as np
from PIL import Image
from .config import REPO
from .geometry import mask_image, resize_mask


class SeemAdapter:
    def __init__(self, settings):
        self.settings = settings
        self.model = None

    def segment(self, image, positive, negative, text):
        has_stroke = positive is not None and positive.any()
        if not has_stroke and not text.strip():
            raise ValueError('Vẽ scribble hoặc nhập mô tả vật thể để SEEM segment.')
        import torch
        from demo_v1 import load_model
        from demo.seem.tasks.interactive import interactive_infer_image
        if not torch.cuda.is_available():
            raise RuntimeError('SEEM demo cần CUDA. Chạy trên RunPod với GPU được bật.')
        if self.model is None:
            for path in [self.settings.seem_checkpoint, self.settings.sam_checkpoint]:
                if not Path(path).is_file():
                    raise FileNotFoundError(f'Thiếu checkpoint: {path}. Chạy setup.sh trước.')
            try:
                self.model, _ = load_model(Namespace(conf_files=self.settings.seem_config,
                    checkpoint=self.settings.seem_checkpoint, sam_checkpoint=self.settings.sam_checkpoint))
            except Exception:
                self.release_gpu()
                raise
        else:
            self.model.cuda()
        # The legacy demo resizes the short side to 512, which can explode the
        # long side on panoramic uploads. Letterbox first and invert the padding.
        factor = 512 / max(image.shape[:2])
        fitted = Image.fromarray(image).resize((max(1, round(image.shape[1]*factor)),
            max(1, round(image.shape[0]*factor))), Image.Resampling.LANCZOS)
        fw, fh = fitted.size
        ox, oy = (512-fw)//2, (512-fh)//2
        model_image = np.zeros((512, 512, 3), dtype=np.uint8)
        model_image[oy:oy+fh, ox:ox+fw] = np.asarray(fitted)

        def fit_mask(mask):
            padded = np.zeros((512, 512), dtype=bool)
            padded[oy:oy+fh, ox:ox+fw] = resize_mask(mask, (fw, fh))
            return mask_image(padded)

        inputs = {'image': Image.fromarray(model_image), 'mask': fit_mask(positive) if has_stroke else None}
        if has_stroke and negative is not None and negative.any():
            inputs['negative_mask'] = fit_mask(negative)
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                _, masks = interactive_infer_image(self.model, None, inputs,
                    ['Stroke'] if has_stroke else ['Text'], reftxt=text, return_masks=True)
            if masks is None or len(masks) == 0:
                raise RuntimeError('SEEM không trả mask. Kiểm tra checkpoint và code SEEM v1.')
            # Single-object UI: select the candidate most covered by the positive stroke.
            candidates = [resize_mask(resize_mask(m, (512, 512))[oy:oy+fh, ox:ox+fw],
                (image.shape[1], image.shape[0])) for m in masks]
            if has_stroke:
                mask = max(candidates, key=lambda m: (m & positive).sum() / max(positive.sum(), 1))
            else:
                mask = candidates[0]
            if not mask.any():
                raise ValueError('Mask SEEM rỗng. Vẽ thêm nét bên trong vật thể hoặc sửa text.')
            mode = 'Stroke (text dùng cho BrushNet)' if has_stroke else 'Text grounding'
            return mask, mode
        finally:
            self.release_gpu()

    def release_gpu(self):
        import torch
        if self.model is not None:
            self.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()


def run_worker(settings, mode, request, job_dir):
    python = settings.depth_python if mode == 'depth' else settings.brushnet_python
    if not Path(python).is_file():
        raise FileNotFoundError(f'Thiếu Python worker: {python}. Chạy bash setup_relocation.sh --skip-seem.')
    request_file = Path(job_dir) / f'{mode}_request.json'
    request_file.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding='utf-8')
    log_file = Path(job_dir) / f'{mode}.log'
    env = os.environ.copy()
    # Never inherit SEEM's Python imports or user site packages into a worker.
    env.pop('PYTHONPATH', None)
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    with log_file.open('w', encoding='utf-8') as output:
        try:
            completed = subprocess.run([str(python), '-m', 'relocation.worker', mode, str(request_file)],
                cwd=str(REPO), env=env, stdout=output, stderr=subprocess.STDOUT,
                timeout=settings.worker_timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f'{mode} vượt timeout {settings.worker_timeout}s. Xem {log_file}') from error
    if completed.returncode:
        tail = '\n'.join(log_file.read_text(encoding='utf-8', errors='replace').splitlines()[-16:])
        raise RuntimeError(f'{mode} worker thất bại. Log: {log_file}\n{tail}')


def repo_revision(path):
    try:
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None
