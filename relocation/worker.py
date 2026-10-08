"""Run with a dedicated Python environment: python -m relocation.worker MODE REQUEST."""
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import (mask_image, dilate, transform_foreground, composite,
                       harmonization_mask, blend_repair)


def gpu_stats(torch):
    torch.cuda.synchronize()
    return {'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
            'gpu': torch.cuda.get_device_name()}


def lama_job(request):
    """Big-LaMa TorchScript protocol used by simple-lama-inpainting.

    Avoid importing the training repo and its old Hydra/Lightning stack.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('LaMa worker cần CUDA.')
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    job = Path(request['job_dir'])
    checkpoint = Path(request['checkpoint'])
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Thiếu Big-LaMa TorchScript: {checkpoint}. Chạy setup_relocation.sh --skip-seem.')
    original = np.array(Image.open(job / 'source.png').convert('RGB'))
    mask = np.array(Image.open(job / 'source_mask.png').convert('L')) > 127
    region = dilate(mask, request['removal_margin'])
    mask_image(region).save(job / 'removal_mask.png')
    h, w = original.shape[:2]
    factor = min(1, request['max_side'] / max(h, w))
    size = (max(1, round(w*factor)), max(1, round(h*factor)))
    image = np.array(Image.fromarray(original).resize(size, Image.Resampling.LANCZOS))
    small_mask = np.array(mask_image(region).resize(size, Image.Resampling.NEAREST)) > 127
    sh, sw = small_mask.shape
    if not small_mask.any():
        raise ValueError('Mask LaMa quá nhỏ sau resize; tăng inference size hoặc mask margin.')
    padding = ((0, (-sh) % 8), (0, (-sw) % 8))
    image = np.pad(image, padding + ((0, 0),), mode='symmetric')
    small_mask = np.pad(small_mask, padding, mode='symmetric')
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255)[None].to('cuda')
    mask_tensor = torch.from_numpy(small_mask.astype(np.float32))[None, None].to('cuda')
    model = torch.jit.load(str(checkpoint), map_location='cuda').eval()
    with torch.inference_mode():
        output = model(image_tensor, mask_tensor)[0].permute(1, 2, 0).cpu().numpy()[:sh, :sw]
    raw = Image.fromarray(np.clip(output*255, 0, 255).astype(np.uint8)).resize((w, h), Image.Resampling.LANCZOS)
    raw.save(job / 'removal_raw.png')
    background = blend_repair(original, np.array(raw), region, feather=2)
    background[mask] = np.array(raw)[mask]
    Image.fromarray(background).save(job / 'background.png')
    metadata = dict(checkpoint=str(checkpoint), model='Big-LaMa TorchScript',
                    seconds=time.time()-started, inference_size=list(size),
                    torch_version=torch.__version__, **gpu_stats(torch))
    (job / 'lama_result.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def depth_job(request):
    import torch
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    model_id = request['model']
    print(f'Loading depth: {model_id}', flush=True)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).eval().to('cuda')
    image = Image.open(request['image']).convert('RGB')
    inputs = processor(images=image, return_tensors='pt').to('cuda')
    with torch.inference_mode():
        result = model(**inputs).predicted_depth
        raw = torch.nn.functional.interpolate(result[:, None].float(),
            size=(image.height, image.width), mode='bicubic', align_corners=False)[0, 0].cpu().numpy()
    np.save(request['output'], raw)
    metric = getattr(model.config, 'depth_estimation_type', 'relative') == 'metric'
    Path(request['metadata']).write_text(json.dumps({'model': model_id,
        'metric': metric, 'dtype': 'float32', 'shape': list(raw.shape),
        'seconds': time.time()-started, **gpu_stats(torch)}, indent=2), encoding='utf-8')


def brushnet_job(request):
    import torch
    import diffusers
    from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline, UniPCMultistepScheduler
    from .adapters import repo_revision
    expected = Path(request['brushnet_repo']).resolve() / 'src'
    if not Path(diffusers.__file__).resolve().is_relative_to(expected):
        raise RuntimeError(f'Diffusers không phải bản BrushNet tại {expected}: {diffusers.__file__}. Cài lại bằng setup_relocation.sh.')
    if not torch.cuda.is_available():
        raise RuntimeError('BrushNet worker cần CUDA.')
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(int(request['seed']))
    torch.cuda.manual_seed_all(int(request['seed']))
    started = time.time()
    job = Path(request['job_dir'])
    image = np.array(Image.open(job / 'source.png').convert('RGB'))
    mask = np.array(Image.open(job / 'source_mask.png').convert('L')) > 127
    rgb, alpha, transform = transform_foreground(image, mask, request['target'],
        request['scale'], request['allow_clipping'])
    mode = request.get('mode', 'generate')
    if mode == 'generate':
        repair_target = dilate(alpha > .01, request['target_margin'])
        protected_core = np.zeros(mask.shape, dtype=bool)
    elif mode == 'preserve':
        repair_target, protected_core = harmonization_mask(alpha, request['target_margin'])
    else:
        raise ValueError(f'Unknown target mode: {mode}')
    mask_image(repair_target).save(job / 'harmonization_mask.png')
    mask_image(repair_target).save(job / 'generation_mask.png')
    Image.fromarray((alpha * 255).astype(np.uint8)).save(job / 'target_mask.png')
    cutout = np.dstack([image, mask.astype(np.uint8) * 255])
    Image.fromarray(cutout).save(job / 'source_cutout.png')
    Image.fromarray(np.dstack([rgb, (alpha * 255).astype(np.uint8)])).save(job / 'target_cutout.png')

    brushnet = BrushNetModel.from_pretrained(request['brushnet_checkpoint'], torch_dtype=torch.float16)
    pipe = StableDiffusionBrushNetPipeline.from_pretrained(request['base_model'],
        brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    h, w = image.shape[:2]
    factor = min(1, request['max_side'] / max(w, h))
    work_size = (max(64, int(w * factor) // 8 * 8), max(64, int(h * factor) // 8 * 8))

    def inpaint(img, region, prompt, negative, seed):
        if not region.any():
            return img.copy()
        cond = Image.fromarray(img).resize(work_size, Image.Resampling.LANCZOS)
        region_work = mask_image(region).resize(work_size, Image.Resampling.NEAREST)
        if not np.asarray(region_work).any():
            raise ValueError('Mask quá nhỏ ở độ phân giải inference. Tăng inference size hoặc mask margin.')
        cond_array = np.asarray(cond).copy()
        cond_array[np.asarray(region_work) > 127] = 0
        out = pipe(prompt=prompt, negative_prompt=negative or None,
            image=Image.fromarray(cond_array), mask=region_work.convert('RGB'),
            width=work_size[0], height=work_size[1],
            generator=torch.Generator('cuda').manual_seed(int(seed)),
            num_inference_steps=int(request['steps']), guidance_scale=float(request['guidance']),
            brushnet_conditioning_scale=float(request['conditioning'])).images[0]
        return np.array(out.resize((w, h), Image.Resampling.LANCZOS))

    background = np.array(Image.open(job / 'background.png').convert('RGB'))
    if background.shape != image.shape:
        raise ValueError('Nền LaMa phải cùng kích thước với ảnh nguồn.')
    pasted = composite(background, rgb, alpha)
    Image.fromarray(pasted).save(job / 'pasted.png')
    print(f'BrushNet target pass: {mode}', flush=True)
    raw_final = inpaint(background if mode == 'generate' else pasted, repair_target, request['target_prompt'],
        request['negative_prompt'], request['seed'])
    Image.fromarray(raw_final).save(job / 'harmonization_raw.png')
    final = blend_repair(background if mode == 'generate' else pasted, raw_final, repair_target, feather=1)
    if mode == 'generate':
        final[alpha > .95] = raw_final[alpha > .95]
    else:
        final[protected_core] = pasted[protected_core]
    Image.fromarray(final).save(job / 'result.png')
    transform.update(seconds=time.time()-started, inference_size=list(work_size),
        brushnet_revision=repo_revision(request['brushnet_repo']),
        torch_version=torch.__version__, diffusers_version=diffusers.__version__,
        mode=mode, **gpu_stats(torch))
    (job / 'brushnet_result.json').write_text(json.dumps(transform, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in {'depth', 'lama', 'brushnet'}:
        raise SystemExit('Usage: python -m relocation.worker {depth|lama|brushnet} request.json')
    request = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
    {'depth': depth_job, 'lama': lama_job, 'brushnet': brushnet_job}[sys.argv[1]](request)


if __name__ == '__main__':
    main()
