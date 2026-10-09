# SEEM + LaMa + BrushNet relocation

Default pipeline: SEEM source segmentation → optional metric depth scale →
Big-LaMa source removal → transformed target mask → BrushNet target generation.
BrushNet source removal is also available for side-by-side comparison, using
the same padded bounding rectangle as LaMa. The precise source silhouette is
retained separately for placement at the destination.
Gradio remains pinned to 3.50.2 to reuse the working SEEM environment.
Vietnamese flow and deployment guide: [FLOW_RUNPOD.md](FLOW_RUNPOD.md).

## Upgrade an existing RunPod

Stop the previous demo with Ctrl+C. Activate the same Python environment used
for demo_v1.py (Python 3.10 recommended).

```bash
cd /workspace/Segment-Everything-Everywhere-All-At-Once
git pull --ff-only origin exp_v1
export BRUSHNET_REPO=/workspace/BrushNet
export HF_HOME=/workspace/hf-cache
export RELOCATION_OUTPUT_DIR=/workspace/relocation_outputs
bash setup_relocation.sh --only-lama
python3 demo_relocation.py --preflight
python3 demo_relocation.py --server-name 0.0.0.0 --port 7860 --max-side 512
```

Expose HTTP port 7860 in RunPod. Hard refresh the browser after restarting.
Existing SEEM/BrushNet/depth weights are reused; --only-lama installs only the
LaMa worker and downloads its checkpoint.

## Fresh RunPod

Use Python 3.10, Ubuntu 22.04, a CUDA devel image for SEEM extension compilation,
one GPU, at least 32 GB host RAM, persistent /workspace storage. Start at 512.

```bash
cd /workspace
git clone --branch exp_v1 https://github.com/KhoaLeDang2375/Segment-Everything-Everywhere-All-At-Once.git
git clone https://github.com/TencentARC/BrushNet.git
cd Segment-Everything-Everywhere-All-At-Once
export BRUSHNET_REPO=/workspace/BrushNet
export HF_HOME=/workspace/hf-cache
export RELOCATION_OUTPUT_DIR=/workspace/relocation_outputs
python3 -m pip install 'numpy<2' 'setuptools<70'
python3 -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
bash setup_relocation.sh
python3 demo_relocation.py --preflight
python3 demo_relocation.py --port 7860 --max-side 512
```

Use --skip-seem when SEEM is already installed, --skip-download to use existing
weights. CLI path overrides include --lama-checkpoint and --lama-python.
LAMA_CHECKPOINT/LAMA_PYTHON environment variables are also supported.

## Model isolation and memory

* Main environment reuses setup.sh and demo_v1.load_model: SEEM v1 + Gradio.
* .venv-lama: Torch 2.1.2/cu121, NumPy 1.26.4, Pillow 9.5.0. Big-LaMa is loaded
  as TorchScript; the old training repo/Hydra/Lightning dependencies are omitted.
* .venv-depth: Transformers 4.46.3 and Depth Anything V2.
* .venv-brushnet: custom Diffusers from the sibling BrushNet repo,
  Transformers 4.38.2, Hub 0.25.2. SD 1.5 checkpoints only, not SDXL.

GPU operations are serialized. SEEM and its language embedding cache are moved
back to CPU after segmentation. Workers exit before the next GPU task.
BrushNet uses FP16, model CPU offload and VAE slicing. 16 GB is a candidate for
512 inference, not a measured guarantee. Adequate CPU RAM is needed for offload.
LaMa/depth/BrushNet report peak allocated/reserved GiB and timings; these are
PyTorch process measurements, not total nvidia-smi device usage.

Big-LaMa export source and protocol:
https://github.com/enesmsahin/simple-lama-inpainting
Setup downloads the project's v0.1.0 big-lama.pt TorchScript release, verifies
it can load and records a local SHA256. It does not claim publisher checksum
verification. Supply a compatible trusted TorchScript export if overriding it.

## Workflow and prompts

1. Upload an image. Working dimensions are capped at 2048 on the longest side.
2. Draw inside the source object, or enter an English description for text-only
   grounding. With strokes, SEEM uses spatial prompting. Optional negative
   scribbles exclude pixels; they are distinct from BrushNet negative text.
3. Segment; inspect the source mask or upload a corrected white-object mask.
4. Select lama/brushnet removal or compare both side by side. Both use a filled
   bounding rectangle with 8 working-image pixels of padding by default.
   BrushNet removal has its own background positive and removal negative text.
   Inspect background.png; adjust padding and rerun if object edges remain.
5. Click the target image to place the mask centroid. Orientation stays fixed.
6. Select Indoor/Outdoor depth correctly (Outdoor is the default), or manually
   choose scale. Inspect the transformed mask/geometry preview.
7. Choose generate (default) or preserve for comparison. Inspect actual prompts.
8. Run and inspect the intermediate images and ZIP metadata.

LaMa uses image + removal mask; it has no positive/negative text prompt.
Generate uses the cleaned background and the ENTIRE transformed mask (plus
small default dilation of 3 px). Masked RGB is zeroed before BrushNet. This is
new synthesis: source appearance/identity and exact silhouette are not guaranteed.
The source RGB cutout/pasted.png are diagnostic comparisons, not conditioning
for generate. Reference-image conditioning is deferred.

Preserve retains the previous RGB cut-and-paste plus ring repair mode for
comparison. It still erodes the protected core slightly; use a narrow margin.
No diffusion pixels escape the recorded generation mask in either mode.

Positive example: A sheep standing naturally on green grass.
Negative example: frame, cage, basket, rope, duplicate objects, artifacts.
Do not enter the desired result in the negative field. The default positive
fallback uses the object description plus coherent lighting/texture wording.
Masks define writable areas, not semantic guarantees.

## Geometry and depth limits

RGB/alpha are transformed together in premultiplied alpha space for previews
and preserve mode. The target is the mask centroid, not the foot contact point.
Clipping requires explicit opt-in. Shadows, occlusion and 3D orientation are
not automatically solved by the 2D transform.
Preview always shows clipped geometry with a status message rather than throwing
an edge-overflow exception. "Thu nhỏ để vừa ảnh" shrinks scale at the fixed
target center. Actual inference still rejects unapproved clipping before model
loading. If scale 0.2 cannot fit at that center, move the center or allow clipping.

Depth is saved as raw float32 NPY. Scale proposal is median source depth divided
by a visible-background depth patch around target, bounded to 0.5–2.0; relative
models only provide previews. Invalid/heterogeneous depth falls back to 1.
Destination background depth is not necessarily the relocated object's center
depth. MAD checks do not measure model confidence. Review scale manually.

## Two reported Gradio failures

* mask=None before the first stroke: SketchImage supplies a blank mask BEFORE
  Gradio Image.preprocess decodes it, fixing None.rsplit for both canvases.
  It retains the image frontend component name for Gradio 3.
* Negative-canvas image mismatch: untouched/empty negative masks are ignored.
  When strokes exist, dimensions and RGB content are checked with small canvas
  round-trip tolerance (mean difference <=3, 99th percentile <=20). Different
  scenes are still rejected; re-upload source if the negative canvas is stale.
* Static-image target clicks: target uses interactive editor because Gradio
  3.50.2 static image binds to a button and sends null coordinates. Coordinates
  are checked for finite values/bounds before conversion.

The Gradio upgrade notice and share=True message are informational. Upgrading
to Gradio 4 would require migrating sketch APIs; no upgrade is needed here.

## Artifacts, validation and codebase index

Each attempt keeps requests, logs, raw results, masks and metadata. Removal
previews are cached per source session/backend/settings (BrushNet includes prompt,
negative, seed and diffusion parameters), then copied into
independent result jobs. Images/JSON/NPY are included in the downloadable ZIP.
Removal provenance is prefixed removal_lama_* or removal_brushnet_* to avoid
being overwritten by the target BrushNet worker. removal_result.json summarizes
backend/mask shape/checkpoint/timing/memory; target_mask retains the silhouette.
Failed workers preserve requests/logs for diagnosis. Outputs require manual
cleanup and persistent storage if they must survive Pod replacement.

RunPod validation remains necessary: source deletion, generate vs preserve,
identity scale, shrink/enlarge, source-target overlap, edge/clipping, empty
strokes, text-only segmentation and repeated runs on 16 GB.
The existing CPU tests describe the initial version; model quality and current
GPU behavior are not established by those checks.

```bash
python -m relocation.index_codebase
```

This refreshes the existing Graphify JSON/HTML with AST symbols/imports and
reviewed integration links. It preserves prior semantic graph provenance;
no Graphify semantic analyzer is installed/run in this workspace.


## BrushNetX source removal and IP-Adapter Plus destination reference

Update an existing pod without reinstalling SEEM/LaMa/depth:

```bash
git pull --ff-only origin exp_v1
bash setup_relocation.sh --only-brushnet
python demo_relocation.py --preflight
python demo_relocation.py --port 7860
```

Source removal uses official TencentARC/BrushEdit `brushnetX` (revision
0d6ac4a), independent from the existing target segmentation checkpoint.
Override via `--removal-brushnet-checkpoint` or `REMOVAL_BRUSHNET_CHECKPOINT`.
LaMa remains the default removal backend. BrushNetX is generative and may
still invent objects: describe the actual background. The grass preset adds
explicit grass positive text and wall/concrete negatives for the sheep example.

Target-only IP-Adapter Plus SD1.5 is enabled by default in the UI, strength 0.6.
Its masked object reference is cropped and square padded on neutral gray;
the source background is excluded. It is passed separately to the UNet image
conditioning, never used for source removal. Toggle it off for an ablation.
Use `IP_ADAPTER_DIR` / `--ip-adapter-dir` for alternative local paths.
FP16 CLIP encoder joins the existing model CPU offload sequence. The custom
Diffusers fork already exposes IPAdapterMixin and `ip_adapter_image`; no stock
Diffusers upgrade is installed. GPU execution/quality/peak memory still requires
validation on RunPod. See FLOW_RUNPOD.md section 10 for commands and details.
