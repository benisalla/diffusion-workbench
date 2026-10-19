"""Three small dataclasses hold every setting: the network, the diffusion method, the training run.

They are saved inside every checkpoint, so `diffusion-sample` and `diffusion-eval` rebuild exactly
the model that was trained, without repeating any flag.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

METHODS = ("ddpm", "score-sde", "flow-matching", "edm")
BACKBONES = ("unet", "dit")


@dataclass
class ModelConfig:
    """The denoising network. Both backbones take (x_t, t, label) and return a tensor shaped like x_t."""

    backbone: str = "unet"          # unet | dit
    in_channels: int = 1            # filled from the dataset
    image_size: int = 28            # filled from the dataset
    num_classes: int = 0            # 0 = unconditional; >0 adds a label embedding (+1 "null" label for CFG)
    base_channels: int = 16         # U-Net width at full resolution (small: trains in minutes on a CPU)
    channel_mults: tuple = (1, 2, 2)  # U-Net width multiplier per resolution (28 -> 14 -> 7)
    dropout: float = 0.0
    dit_dim: int = 128              # DiT token width
    dit_depth: int = 4              # DiT blocks
    dit_heads: int = 4
    patch_size: int = 4             # DiT patch size: 28 px / 4 = 7 x 7 = 49 tokens


@dataclass
class MethodConfig:
    """Which diffusion / flow formulation is trained, and its few hyper-parameters."""

    method: str = "ddpm"            # ddpm | score-sde | flow-matching | edm
    prediction: str = "eps"         # ddpm only: eps | x0 | v
    schedule: str = "linear"        # ddpm only: linear | cosine
    timesteps: int = 1000           # ddpm only: T
    sde: str = "vp"                 # score-sde only: vp | ve


@dataclass
class TrainConfig:
    dataset: str = "mnist"          # mnist | fashion-mnist | cifar10 | shapes
    data_dir: str = "data"
    n_train: int = 0                # 0 = whole training set, else a random subset (fast tests)
    cond: bool = False              # train class-conditional (needed for classifier-free guidance)
    p_uncond: float = 0.1           # label dropout probability when cond=True
    batch_size: int = 64
    lr: float = 1e-3
    warmup_steps: int = 100
    max_steps: int = 2000
    max_minutes: float = 0.0        # 0 = no time limit, else stop after this many minutes
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    log_every: int = 50
    seed: int = 0
    threads: int = 1
    device: str = "auto"            # auto | cpu | cuda
    output_dir: str = "checkpoints/run"


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RunConfig:
        def build(klass, values):
            names = {f.name for f in fields(klass)}
            kw = {k: v for k, v in values.items() if k in names}
            if "channel_mults" in kw:
                kw["channel_mults"] = tuple(kw["channel_mults"])
            return klass(**kw)

        return cls(build(ModelConfig, d.get("model", {})), build(MethodConfig, d.get("method", {})),
                   build(TrainConfig, d.get("train", {})))
