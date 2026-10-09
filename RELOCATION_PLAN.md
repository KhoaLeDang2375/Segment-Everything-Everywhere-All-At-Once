# Relocation revision plan

## Implemented in this revision

1. Reuse SEEM setup/loader, preserve stroke/text routing and centroid target.
2. Add Big-LaMa TorchScript source-removal worker in a separate Python environment.
3. Expose removal preview before target synthesis; cache by source session/margin.
4. Default BrushNet mode generates the entire transformed target mask, with small
   dilation. Keep RGB paste/ring repair as preserve comparison mode.
5. Separate positive and negative text clearly, show effective prompts; LaMa
   requires no text. Default outdoor depth with explicit indoor choice.
6. Serialize GPU tasks, release model/cache memory, record worker peaks/timings.
7. Add --only-lama upgrade path, checkpoint download and preflight paths/imports.
8. Fix both pasted-log failures: blank sketch mask is supplied before Gradio
   preprocessing; empty negative strokes skip source comparison and active
   strokes use dimension/content checks with browser round-trip tolerance.
9. Update RunPod instructions and the existing Graphify AST supplement.
10. Restore BrushNet source removal as an alternative to LaMa. Both use a padded
    bounding rectangle without object silhouette; exact mask stays for target.
11. Add side-by-side removal comparison, independent removal/target prompts,
    backend/settings-specific caches and prefixed worker provenance in ZIPs.
12. Provide Vietnamese flow and deployment commands in FLOW_RUNPOD.md.
13. Show clipped previews without exceptions, offer shrink-to-fit at the fixed
    center, and report geometry validation in UI status before inference.
14. Separate removal BrushNetX checkpoint from segmentation checkpoint at target;
    provide grass background prompt preset and preserve LaMa comparison.
15. Add optional target-only IP-Adapter Plus SD1.5 with masked square reference,
    strength control, FP16 encoder, optional CPU offload and ZIP provenance.
16. Add --only-brushnet upgrade/download path; keep existing worker versions.

17. Default BrushNet to full FP16 CUDA for RTX 4000 Ada 20 GB; keep CLI/env
    cpu-offload option, per-stage timings and actionable OOM messages. Workers
    remain short-lived and GPU operations stay serialized.

## Validation still required on RunPod

* Full CUDA vs CPU offload timings/peaks on RTX 4000 Ada 20 GB at 512.
* Real LaMa checkpoint inference and background quality for the uploaded sheep.
* LaMa vs BrushNet source removal with the same rectangular mask and padding.
* BrushNetX background quality and IP-Adapter Plus integration/identity with
  the custom Diffusers fork; compare adapter on/off at fixed seed and inspect peaks.
* Full target generation vs preserve; inspect artifacts and exact effective prompts.
* Empty positive/negative strokes, text-only, active negative strokes, stale image.
* Target clicks, overlapping source/target, scaling and clipping at boundaries.
* Repeated 512 inference on RTX 2000 Ada 16 GB; inspect both worker memory logs
  and total device memory. No GPU quality/VRAM guarantee is inferred from syntax.

## Deferred

3D occlusion, ground-contact reasoning, shadows and orientation changes.
The mask constrains editable pixels, not exact generated silhouette or identity.
