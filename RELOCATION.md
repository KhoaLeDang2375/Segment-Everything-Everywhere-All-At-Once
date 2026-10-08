# SEEM v1 + BrushNet relocation demo

Gradio 3.50.2, one uploaded image and one selected object per run. The target
point is the center of the mask. The object retains its orientation. Source RGB
and alpha are transformed together; BrushNet fills the source region, then
inpaints a boundary ring at the destination while preserving the pasted RGB core.

## RunPod setup

Use the same Python environment that successfully ran `demo_v1.py` (recommended
Python 3.10, CUDA). Keep BrushNet beside this repository, or set `BRUSHNET_REPO`.
Stop a running `demo_v1.py` before launching relocation so its GPU allocations
and port 7860 are released.

```bash
cd /workspace/Segment-Everything-Everywhere-All-At-Once
export BRUSHNET_REPO=/workspace/BrushNet
# Optional: point HF_HOME and RELOCATION_OUTPUT_DIR at a persistent volume.
export HF_HOME=/workspace/hf-cache
export RELOCATION_OUTPUT_DIR=/workspace/relocation_outputs
bash setup_relocation.sh --skip-seem
python demo_relocation.py --preflight
python demo_relocation.py --port 7860
```

On a fresh pod, omit `--skip-seem` to reuse the existing `setup.sh` first. To reuse
downloaded weights, add `--skip-download` and pass existing paths:

```bash
python demo_relocation.py \
  --base-model /workspace/models/realisticVisionV60B1_v51VAE \
  --brushnet-checkpoint /workspace/models/segmentation_mask_brushnet_ckpt \
  --max-side 512 --port 7860
```

The base model must be SD 1.5 compatible and saved in Diffusers directory format;
the BrushNet checkpoint must match SD 1.5. SDXL is not used by this entry point.
Weights default to the official TencentARC BrushNet Space, using the same base
and BrushNet checkpoints as the repository's working inference example:
https://huggingface.co/spaces/TencentARC/BrushNet/tree/main/data/ckpt

RunPod should expose HTTP port 7860. `--share` additionally creates a Gradio share
link only when requested. All jobs/images remain in `RELOCATION_OUTPUT_DIR` until
you remove them; ZIP downloads contain source, masks, intermediate images and
metadata. Code/checkpoints/outputs should live on persistent storage if needed
after a pod is stopped.

## Dependency isolation and VRAM

* Main environment: reuse `setup.sh` / `demo_v1.py` with SEEM's Transformers 4.34
  and Gradio 3.50.2.
* `.venv-brushnet`: local BrushNet fork of Diffusers, Transformers 4.38.2, Hub
  0.25.2 (the fork still imports `cached_download`). Stock PyPI Diffusers is rejected.
* `.venv-depth`: Transformers 4.46.3, providing Depth Anything V2.
* Each worker exits after inference, releasing its CUDA context. SEEM is moved to
  CPU after mask prediction. GPU tasks use a shared lock; Gradio queues one event
  at a time. SEEM stays in CPU RAM for later segmentation.
* Worker torch/torchvision are pinned to 2.1.2/0.16.2 cu121. Worker locks are recorded
  as `.venv-*/installed.txt` after setup. Packages are isolated from the main environment.

Default diffusion long side is 512, selectable via `--max-side 768` or `1024`.
CPU offload is enabled for BrushNet. Plan on adequate CPU RAM for resident SEEM
weights and diffusion offload (prefer a pod with at least 32 GB system RAM).
A4500 has 20 GB VRAM, A5000 has 24 GB. Actual CUDA memory/latency must be measured
on your pod; no GPU inference is claimed by the local CPU checks.

## User workflow

1. Upload an image. A working copy is capped at 2048 pixels on its longest side;
   masks, coordinates and exported results use that working image.
2. Draw a positive stroke **inside** the object. Optionally draw negative strokes
   on the second canvas to exclude unwanted regions, and segment again.
   SEEM inputs are letterboxed to 512×512, then predicted masks are unpadded and
   restored to working-image coordinates; panoramic uploads cannot expand the
   SEEM input to an arbitrarily large long side.
3. Enter an English object description. With a scribble, SEEM v1 uses the spatial
   route; text is used in BrushNet prompts. Without a scribble, text grounding
   selects the source mask. The updated SEEM v1 spatial decoder takes precedence
   over text; this demo does not claim simultaneous spatial/text fusion.
4. Check the red mask overlay. A corrected black/white mask can be uploaded
   instead. A closed outline around empty background is not the same prompt as
   a stroke on the object: fill/draw inside the object for spatial segmentation.
5. Click the target preview. The generated arrow points from source mask centroid
   to target centroid. Arrow pixels never enter SEEM's scribble prompt.
6. Optionally estimate depth scale, then adjust the scale slider and preview.
7. Describe the background to restore, adjust the target prompt if necessary,
   and run relocation. The second pass edits only the boundary ring, not the
   entire target object interior.

The legacy VIBE notebook's combined red-outline/arrow PNG parser is not reused
for live annotations: direct click coordinates are explicit and avoid skeleton
endpoint ambiguity. Uploaded source images, manually corrected masks and the
depth logic are supported; VIBE dataset batch loading is a later extension.

## Depth and geometry

Metric indoor/outdoor checkpoints return raw floating point distance estimates.
Source depth is the median inside an eroded object mask; destination depth is
the median of a small patch around the target, excluding pixels covered by the
source object. If no visible background remains in that patch, scale defaults
to 1.0 and should be adjusted manually. Scale is `Z_source / Z_target`,
limited to 0.5–2 for automatic proposals. Invalid/heterogeneous regional depth
falls back to 1.0. The slider also permits 0.2–3.0 with explicit user control.

Relative depth is saved and visualized but never used as a physically meaningful
ratio. Display normalization is only for the PNG preview; raw depth remains in
`depth.npy`. A regional MAD check detects some depth variation; it is not a model
confidence guarantee or a metric-depth accuracy estimate.

RGB is warped in premultiplied alpha space to avoid dark fringes. The actual
transformed silhouette is clipped by the image canvas, never by clamping every
contour vertex. Clipping requires the user checkbox. Inpainting output is pasted
only in its recorded repair region; protected object interior and other pixels
are retained. Source mask margin can remove some old shadow, but automatic
shadow segmentation, new shadow generation, depth-aware occlusion, physical
surface contact and relighting are not guaranteed. This is a 2D relocation baseline.

## Codebase index and checks

```bash
python -m relocation.index_codebase
python -m unittest relocation.test_geometry -v
python -m unittest relocation.test_controller -v
```

`graphify-out/relocation-index.json` stores file hashes, symbol lines, imports and
reviewed integration links. `graphify-out/graph.json` is incrementally extended
with AST nodes for this app. The earlier semantic graph provenance is retained;
`graph.html` displays the same added integration community while keeping its
existing viewer and earlier graph data. No Graphify executable
is installed locally, so the update is a deterministic AST supplement, not a
claim that Graphify's semantic analyzer was rerun.

`--preflight` checks checkpoint paths, imports and CUDA availability. To validate
real inference on RunPod, use one source object and test identity scale, shrinking,
enlarging, source/target overlap and an edge target. Inspect all intermediate
images and `metadata.json`, not only the final image. Each attempt has a separate
directory; failed worker logs/requests remain there for diagnosis.
