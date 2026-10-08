"""Per-session image state, disk artifacts, and serial GPU scheduling."""
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone
import json
import threading
import uuid
import zipfile
import numpy as np
from PIL import Image
from .config import REPO, DEPTH_MODELS
from .adapters import SeemAdapter, run_worker, repo_revision
from .geometry import (ink, image_hash, mask_image, resize_mask, centroid,
                       suggest_scale, transform_foreground, composite)


def uploaded_image(value):
    if value is None:
        raise ValueError('Upload ảnh trước.')
    value = value.get('image') if isinstance(value, dict) else value
    if value is None:
        raise ValueError('Canvas không có ảnh.')
    return np.asarray(Image.fromarray(np.asarray(value, dtype=np.uint8)).convert('RGB'))


def working_image(value):
    raw = uploaded_image(value)
    pil = Image.fromarray(raw)
    pil.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    return raw, np.array(pil)


class Controller:
    def __init__(self, settings):
        self.settings = settings
        self.seem = SeemAdapter(settings)
        self.gpu_lock = threading.Lock()

    def validate(self, canvas, state):
        if not state:
            raise ValueError('Segment hoặc upload mask trước khi tiếp tục.')
        if image_hash(uploaded_image(canvas)) != state['image_hash']:
            raise ValueError('Ảnh đã thay đổi. Segment lại để tránh sử dụng mask của ảnh cũ.')
        if state.get('scribble_hash') is not None:
            current = ink(canvas.get('mask')) if isinstance(canvas, dict) else None
            if current is None:
                current = np.zeros(uploaded_image(canvas).shape[:2], dtype=bool)
            if image_hash(current) != state['scribble_hash']:
                raise ValueError('Scribble đã thay đổi. Bấm lấy mask lại trước khi di chuyển.')

    def segment(self, canvas, negative_canvas, text):
        raw, image = working_image(canvas)
        positive = ink(canvas.get('mask')) if isinstance(canvas, dict) else None
        negative = ink(negative_canvas.get('mask')) if isinstance(negative_canvas, dict) else None
        size = (image.shape[1], image.shape[0])
        if positive is not None:
            positive = resize_mask(positive, size)
        if negative is not None and negative.any():
            negative_image = uploaded_image(negative_canvas)
            # Gradio's browser canvas round trip can alter RGB values slightly.
            # Compare content with a small tolerance, never accept another scene.
            same = negative_image.shape == raw.shape
            if same:
                difference = np.abs(negative_image.astype(np.float32) - raw.astype(np.float32))
                same = float(difference.mean()) <= 3 and float(np.percentile(difference, 99)) <= 20
            if not same:
                raise ValueError('Canvas negative scribble phải sử dụng cùng ảnh nguồn.')
            negative = resize_mask(negative, size)
        else:
            negative = None
        with self.gpu_lock:
            mask, mode = self.seem.segment(image, positive, negative, text)
        state = self.new_state(raw, image, mask, mode)
        state['seem_metrics'] = getattr(self.seem, 'metrics', None)
        raw_positive = ink(canvas.get('mask')) if isinstance(canvas, dict) else None
        state['scribble_hash'] = image_hash(raw_positive if raw_positive is not None else np.zeros(raw.shape[:2], dtype=bool))
        if positive is not None:
            mask_image(positive).save(Path(state['job']) / 'positive_scribble.png')
        if negative is not None:
            mask_image(negative).save(Path(state['job']) / 'negative_scribble.png')
        return state

    def new_state(self, raw, image, mask, mode):
        centroid(mask)
        job = Path(self.settings.output_dir).resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:10])
        job.mkdir(parents=True, exist_ok=False)
        Image.fromarray(image).save(job / 'source.png')
        mask_image(mask).save(job / 'source_mask.png')
        return {'image_hash': image_hash(raw), 'image': image, 'mask': mask,
                'job': str(job), 'segmentation_mode': mode, 'depth': None}

    def import_mask(self, canvas, uploaded):
        raw, image = working_image(canvas)
        if uploaded is None:
            raise ValueError('Upload mask đen/trắng (trắng = vật thể) trước.')
        arr = np.asarray(uploaded)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=-1)
        mask = resize_mask(arr > 127, (image.shape[1], image.shape[0]))
        return self.new_state(raw, image, mask, 'Uploaded/corrected mask')

    def estimate(self, canvas, state, target, mode):
        self.validate(canvas, state)
        if target is None:
            raise ValueError('Bấm lên ảnh chọn tâm đích trước.')
        if mode not in DEPTH_MODELS:
            raise ValueError('Chọn mô hình depth trước khi ước lượng.')
        model = DEPTH_MODELS[mode]
        job = Path(state['job'])
        with self.gpu_lock:
            if state.get('depth_model') != model or not (job / 'depth.npy').exists():
                request = {'image': str(job / 'source.png'), 'model': model,
                    'output': str(job / 'depth.npy'), 'metadata': str(job / 'depth_model.json')}
                run_worker(self.settings, 'depth', request, job)
            raw = np.load(job / 'depth.npy', allow_pickle=False)
            model_info = json.loads((job / 'depth_model.json').read_text(encoding='utf-8'))
            info = suggest_scale(raw, state['mask'], target, is_metric=model_info['metric'])
        state = dict(state, depth_model=model, depth=info, depth_target=list(target))
        finite = raw[np.isfinite(raw)]
        if finite.size:
            lo, hi = np.percentile(finite, [2, 98])
            displayed = np.clip((raw - lo) / max(float(hi-lo), 1e-6), 0, 1)
            displayed = np.nan_to_num(displayed)
        else:
            displayed = np.zeros(raw.shape)
        Image.fromarray((displayed * 255).astype(np.uint8)).save(job / 'depth_preview.png')
        return state, info, Image.open(job / 'depth_preview.png').copy()

    def preview(self, canvas, state, target, scale, allow_clipping):
        self.validate(canvas, state)
        if target is None:
            raise ValueError('Chọn tâm đích trước.')
        rgb, alpha, _ = transform_foreground(state['image'], state['mask'], target, scale, allow_clipping)
        background = state['image']
        if state.get('removal'):
            background = np.array(Image.open(Path(state['removal']['job']) / 'background.png').convert('RGB'))
        return Image.fromarray(composite(background, rgb, alpha)), Image.fromarray((alpha*255).astype(np.uint8))

    def remove(self, canvas, state, removal_margin):
        """Preview/cache LaMa removal independently of target geometry/prompts."""
        self.validate(canvas, state)
        margin = int(removal_margin)
        cached = state.get('removal')
        if cached and cached['margin'] == margin and cached['checkpoint'] == self.settings.lama_checkpoint and Path(cached['job'], 'background.png').is_file():
            return state, Image.open(Path(cached['job']) / 'background.png').copy()
        job = Path(state['job']) / ('removal_' + uuid.uuid4().hex[:10])
        job.mkdir()
        import shutil
        for name in ['source.png', 'source_mask.png']:
            shutil.copy2(Path(state['job']) / name, job / name)
        request = {'job_dir': str(job), 'checkpoint': self.settings.lama_checkpoint,
                   'removal_margin': margin, 'max_side': self.settings.max_side}
        with self.gpu_lock:
            self.seem.release_gpu()
            run_worker(self.settings, 'lama', request, job)
        state = dict(state, removal={'job': str(job), 'margin': margin,
                                   'checkpoint': self.settings.lama_checkpoint})
        return state, Image.open(job / 'background.png').copy()

    def relocate(self, canvas, state, target, scale, text, background_prompt, target_prompt,
                 negative_prompt, steps, guidance, conditioning, seed, removal_margin, target_margin, allow_clipping,
                 mode='generate'):
        self.validate(canvas, state)
        if target is None:
            raise ValueError('Chọn tâm đích trước.')
        if not text.strip():
            raise ValueError('Nhập tên/mô tả vật thể để tạo prompt phù hợp.')
        if mode not in {'generate', 'preserve'}:
            raise ValueError('Chọn chế độ sinh đích hợp lệ.')
        # Validate geometry before allocating/loading the diffusion model.
        transform_foreground(state['image'], state['mask'], target, scale, allow_clipping)
        state, _ = self.remove(canvas, state, removal_margin)
        parent = Path(state['job'])
        job = parent / ('run_' + uuid.uuid4().hex[:10])
        job.mkdir()
        for name in ['source.png', 'source_mask.png', 'positive_scribble.png', 'negative_scribble.png', 'depth.npy', 'depth_preview.png', 'depth_model.json']:
            if (parent / name).is_file():
                import shutil
                shutil.copy2(parent / name, job / name)
        import shutil
        for name in ['background.png', 'removal_mask.png', 'removal_raw.png', 'lama_result.json', 'lama_request.json']:
            shutil.copy2(Path(state['removal']['job']) / name, job / name)
        prompt = target_prompt.strip() or f'{text.strip()}, naturally integrated with the surrounding scene, coherent lighting and texture.'
        request = {'job_dir': str(job), 'brushnet_repo': self.settings.brushnet_repo,
            'base_model': self.settings.base_model, 'brushnet_checkpoint': self.settings.brushnet_checkpoint,
            'target': list(target), 'scale': float(scale), 'allow_clipping': bool(allow_clipping),
            'target_prompt': prompt,
            'negative_prompt': negative_prompt.strip(),
            'steps': int(steps), 'guidance': float(guidance), 'conditioning': float(conditioning),
            'seed': int(seed), 'removal_margin': int(removal_margin), 'target_margin': int(target_margin),
            'max_side': self.settings.max_side, 'mode': mode}
        with self.gpu_lock:
            run_worker(self.settings, 'brushnet', request, job)
        result_info = json.loads((job / 'brushnet_result.json').read_text(encoding='utf-8'))
        depth_info = state.get('depth') if state.get('depth_target') == list(target) else None
        metadata = {'created_at': datetime.now(timezone.utc).isoformat(),
            'seem_revision': repo_revision(REPO), 'settings': asdict(self.settings),
            'input_hash': state['image_hash'], 'segmentation_mode': state['segmentation_mode'],
            'depth_estimate': depth_info, 'seem_metrics': state.get('seem_metrics'),
            'removal': json.loads((job / 'lama_result.json').read_text(encoding='utf-8')),
            'object_text': text, 'request': request, 'result': result_info}
        (job / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        archive = job / 'relocation.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as out:
            for artifact in job.iterdir():
                if artifact.suffix in {'.png', '.json', '.npy'}:
                    out.write(artifact, artifact.name)
        gallery = [(Image.open(job / name).copy(), caption) for name, caption in [
            ('source_mask.png', 'Mask nguồn'), ('removal_mask.png', 'Vùng xóa'),
            ('background.png', 'Nền sau xóa'), ('target_cutout.png', 'Cutout đích'),
            ('pasted.png', 'Mốc so sánh cắt–dán'), ('target_mask.png', 'Mask đích'),
            ('generation_mask.png', 'Vùng BrushNet sinh lại'), ('harmonization_raw.png', 'BrushNet raw')]]
        return Image.open(job / 'result.png').copy(), gallery, str(archive), str(job)
