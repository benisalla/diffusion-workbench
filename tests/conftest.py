import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pytest
import torch

torch.set_num_threads(1)

from diffusion.config import ModelConfig


@pytest.fixture
def small_cfg():
    """A very small network so every test runs in a fraction of a second."""
    return lambda backbone="unet", num_classes=0: ModelConfig(
        backbone=backbone, base_channels=8, channel_mults=(1, 2), num_classes=num_classes,
        dit_dim=32, dit_depth=1, dit_heads=2, patch_size=7,
    )
