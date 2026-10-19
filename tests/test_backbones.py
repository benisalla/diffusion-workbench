import pytest
import torch

from diffusion.backbones import build_backbone, count_parameters, timestep_embedding


@pytest.mark.parametrize("backbone", ["unet", "dit"])
@pytest.mark.parametrize("num_classes", [0, 10])
def test_output_shape_and_zero_init(small_cfg, backbone, num_classes):
    net = build_backbone(small_cfg(backbone, num_classes))
    x = torch.randn(3, 1, 28, 28)
    y = torch.tensor([0, 5, 10]) if num_classes else None  # 10 = the null label
    out = net(x, torch.rand(3) * 999, y)
    assert out.shape == x.shape
    # the output layer is zero-initialised: an untrained network predicts exactly 0
    assert out.abs().max() == 0


def test_default_sizes_are_small():
    from diffusion.config import ModelConfig

    assert count_parameters(build_backbone(ModelConfig())) < 400_000
    assert count_parameters(build_backbone(ModelConfig(backbone="dit"))) < 1_500_000


def test_label_changes_output_after_training_step(small_cfg):
    net = build_backbone(small_cfg("unet", 4))
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    x, t = torch.randn(4, 1, 28, 28), torch.rand(4) * 999
    loss = (net(x, t, torch.arange(4)) - 1).pow(2).mean()
    loss.backward()
    opt.step()
    a = net(x[:1], t[:1], torch.tensor([0]))
    b = net(x[:1], t[:1], torch.tensor([3]))
    assert not torch.allclose(a, b)


def test_timestep_embedding_distinguishes_times():
    e = timestep_embedding(torch.tensor([0.0, 1.0, 500.0]), 32)
    assert e.shape == (3, 32)
    assert (e[0] - e[1]).abs().sum() > 0.1 and (e[1] - e[2]).abs().sum() > 0.1


def test_dit_rejects_bad_patch_size(small_cfg):
    cfg = small_cfg("dit")
    cfg.patch_size = 5
    with pytest.raises(ValueError):
        build_backbone(cfg)
