"""Paths resolve against the repository, independently of the launch directory."""
from dataclasses import dataclass
from pathlib import Path
import os

REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv('BRUSHNET_MODEL_DIR', str(REPO / 'checkpoints/brushnet')))


@dataclass
class Settings:
    seem_config: str = str(REPO / 'configs/seem/samvitl_unicl_lang_v1.yaml')
    seem_checkpoint: str = str(REPO / 'checkpoints/seem_samvitl_v1.pt')
    sam_checkpoint: str = str(REPO / 'checkpoints/sam_vit_l_0b3195.pth')
    brushnet_repo: str = os.getenv('BRUSHNET_REPO', str(REPO.parent / 'BrushNet'))
    brushnet_python: str = os.getenv('BRUSHNET_PYTHON', str(REPO / '.venv-brushnet/bin/python'))
    depth_python: str = os.getenv('DEPTH_PYTHON', str(REPO / '.venv-depth/bin/python'))
    lama_python: str = os.getenv('LAMA_PYTHON', str(REPO / '.venv-lama/bin/python'))
    lama_checkpoint: str = os.getenv('LAMA_CHECKPOINT', str(REPO / 'checkpoints/lama/big-lama.pt'))
    base_model: str = os.getenv('BRUSHNET_BASE_MODEL', str(MODEL_DIR / 'realisticVisionV60B1_v51VAE'))
    brushnet_checkpoint: str = os.getenv('BRUSHNET_CHECKPOINT', str(MODEL_DIR / 'segmentation_mask_brushnet_ckpt'))
    removal_brushnet_checkpoint: str = os.getenv('REMOVAL_BRUSHNET_CHECKPOINT', str(MODEL_DIR / 'brushnetX'))
    ip_adapter_dir: str = os.getenv('IP_ADAPTER_DIR', str(REPO / 'checkpoints/ip-adapter'))
    output_dir: str = os.getenv('RELOCATION_OUTPUT_DIR', str(REPO / 'relocation_outputs'))
    max_side: int = 512
    worker_timeout: int = 1800


DEPTH_MODELS = {
    'Metric indoor': 'depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf',
    'Metric outdoor': 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf',
    'Relative (preview only)': 'depth-anything/Depth-Anything-V2-Base-hf',
}
