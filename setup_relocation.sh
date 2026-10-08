#!/usr/bin/env bash
# Reuse the known-working SEEM setup; add isolated BrushNet/depth environments.
set -euo pipefail
RELOCATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$RELOCATION_ROOT"
SKIP_SEEM=0
SKIP_DOWNLOAD=0
for argument in "$@"; do
    case "$argument" in
        --skip-seem) SKIP_SEEM=1 ;;
        --skip-download) SKIP_DOWNLOAD=1 ;;
        *) echo "Usage: bash setup_relocation.sh [--skip-seem] [--skip-download]"; exit 2 ;;
    esac
done
python3 -c 'import sys; assert sys.version_info[:2] in [(3,9),(3,10),(3,11)], "Use Python 3.9–3.11 (recommended: existing SEEM Python 3.10)"'
if [[ "$SKIP_SEEM" == 0 ]]; then
    bash setup.sh
fi
BRUSHNET_REPO="${BRUSHNET_REPO:-$(dirname "$RELOCATION_ROOT")/BrushNet}"
if [[ ! -f "$BRUSHNET_REPO/src/diffusers/models/brushnet.py" ]]; then
    echo "BrushNet source missing at $BRUSHNET_REPO. Set BRUSHNET_REPO to your clone."; exit 1
fi
for worker in brushnet depth; do
    worker_env="$RELOCATION_ROOT/.venv-$worker"
    if [[ ! -x "$worker_env/bin/python" ]]; then
        python3 -m venv "$worker_env"
    fi
    "$worker_env/bin/python" -m pip install 'pip<25' 'setuptools<70' wheel
    "$worker_env/bin/python" -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
    "$worker_env/bin/python" -m pip install -r "relocation/requirements-$worker.txt"
    if [[ "$worker" == brushnet ]]; then
        # Install this repo's modified Diffusers, never the stock PyPI package.
        "$worker_env/bin/python" -m pip install --no-deps -e "$BRUSHNET_REPO"
    fi
    "$worker_env/bin/python" -m pip check
    "$worker_env/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in worker environment"; print(torch.__version__, torch.cuda.get_device_name())'
    "$worker_env/bin/python" -m pip freeze > "$worker_env/installed.txt"
done
if [[ "$SKIP_DOWNLOAD" == 0 ]]; then
    BRUSHNET_DEST="${BRUSHNET_MODEL_DIR:-$RELOCATION_ROOT/checkpoints/brushnet}"
    export BRUSHNET_DEST
    .venv-brushnet/bin/python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil
destination = Path(os.environ['BRUSHNET_DEST'])
names = ['segmentation_mask_brushnet_ckpt', 'realisticVisionV60B1_v51VAE']
pending = [name for name in names if not (destination / name / '.download_complete').exists()]
if pending:
    snapshot_download(repo_id='TencentARC/BrushNet', repo_type='space',
        allow_patterns=[f'data/ckpt/{name}/**' for name in pending],
        local_dir=str(destination / 'download'), local_dir_use_symlinks=False)
for name in pending:
    source = destination / 'download/data/ckpt' / name
    if not source.is_dir():
        raise RuntimeError(f'Download did not produce {source}')
    target = destination / name
    if target.exists():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.move(str(source), str(target))
    (target / '.download_complete').touch()
PY
    .venv-depth/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf')
PY
fi
printf '%s\n' 'Setup complete. Launch with the SAME Python environment used for demo_v1.py:' \
    'python demo_relocation.py --preflight' \
    'python demo_relocation.py --port 7860'
