"""Run with a dedicated Python environment: python -m relocation.worker MODE REQUEST."""
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import (mask_image, dilate, transform_foreground, composite,
                       harmonization_mask, blend_repair)


def depth_job(request):
    import torch
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
        'metric': metric, 'dtype': 'float32', 'shape': list(raw.shape)}, indent=2), encoding='utf-8')


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
    started = time.time()
    job = Path(request['job_dir'])
    image = np.array(Image.open(job / 'source.png').convert('RGB'))
    mask = np.array(Image.open(job / 'source_mask.png').convert('L')) > 127
    rgb, alpha, transform = transform_foreground(image, mask, request['target'],
        request['scale'], request['allow_clipping'])
    repair_source = dilate(mask, request['removal_margin'])
    repair_target, protected_core = harmonization_mask(alpha, request['target_margin'])
    mask_image(repair_source).save(job / 'removal_mask.png')
    mask_image(repair_target).save(job / 'harmonization_mask.png')
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

    print('BrushNet pass 1: removal', flush=True)
    raw_removed = inpaint(image, repair_source, request['background_prompt'],
        request['removal_negative'], request['seed'])
    Image.fromarray(raw_removed).save(job / 'removal_raw.png')
    # Source object must be fully removed even when feathering the region edges.
    background = blend_repair(image, raw_removed, repair_source, feather=2)
    background[mask] = raw_removed[mask]
    Image.fromarray(background).save(job / 'background.png')
    pasted = composite(background, rgb, alpha)
    Image.fromarray(pasted).save(job / 'pasted.png')
    print('BrushNet pass 2: boundary harmonization', flush=True)
    raw_final = inpaint(pasted, repair_target, request['target_prompt'],
        request['negative_prompt'], request['seed'] + 1)
    Image.fromarray(raw_final).save(job / 'harmonization_raw.png')
    final = blend_repair(pasted, raw_final, repair_target, feather=1)
    final[protected_core] = pasted[protected_core]
    Image.fromarray(final).save(job / 'result.png')
    transform.update(seconds=time.time()-started, inference_size=list(work_size),
        brushnet_revision=repo_revision(request['brushnet_repo']),
        torch_version=torch.__version__, diffusers_version=diffusers.__version__,
        gpu=torch.cuda.get_device_name(), mode='preserve RGB core; inpaint boundary ring')
    (job / 'brushnet_result.json').write_text(json.dumps(transform, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in {'depth', 'brushnet'}:
        raise SystemExit('Usage: python -m relocation.worker {depth|brushnet} request.json')
    request = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
    (depth_job if sys.argv[1] == 'depth' else brushnet_job)(request)


if __name__ == '__main__':
    main()
