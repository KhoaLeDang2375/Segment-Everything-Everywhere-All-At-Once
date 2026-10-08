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

## Validation still required on RunPod

* Real LaMa checkpoint inference and background quality for the uploaded sheep.
* Full target generation vs preserve; inspect artifacts and exact effective prompts.
* Empty positive/negative strokes, text-only, active negative strokes, stale image.
* Target clicks, overlapping source/target, scaling and clipping at boundaries.
* Repeated 512 inference on RTX 2000 Ada 16 GB; inspect both worker memory logs
  and total device memory. No GPU quality/VRAM guarantee is inferred from syntax.

## Deferred

Reference-image conditioning to preserve source identity in generate mode;
3D occlusion, ground-contact reasoning, shadows and orientation changes.
The mask constrains editable pixels, not exact generated silhouette or identity.
