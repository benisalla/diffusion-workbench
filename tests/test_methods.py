import math

import pytest
import torch

from diffusion.backbones import build_backbone
from diffusion.config import MethodConfig
from diffusion.methods import DDPM, EDM, FlowMatching, ScoreSDE, build_method, guided, method_label
from diffusion.sample import NFECounter, generate

ALL = [
    MethodConfig("ddpm", "eps", "linear"), MethodConfig("ddpm", "x0", "cosine"), MethodConfig("ddpm", "v", "cosine"),
    MethodConfig("score-sde", sde="vp"), MethodConfig("score-sde", sde="ve"),
    MethodConfig("flow-matching"), MethodConfig("edm"),
]


@pytest.mark.parametrize("mc", ALL, ids=method_label)
def test_loss_is_finite_and_backpropagates(small_cfg, mc):
    net = build_backbone(small_cfg("unet", 3))
    loss = build_method(mc).loss(net, torch.rand(4, 1, 28, 28) * 2 - 1, torch.tensor([0, 1, 2, 3]))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad)


@pytest.mark.parametrize("mc", ALL, ids=method_label)
def test_every_sampler_runs_and_counts_nfe(small_cfg, mc):
    method = build_method(mc)
    net = NFECounter(build_backbone(small_cfg("unet")).eval())
    for sampler in method.samplers:
        for nfe in (1, 5, 6):
            net.calls = 0
            steps = method.steps_for_nfe(sampler, nfe)
            x = method.sample(net, 2, steps, sampler, shape=(1, 28, 28), generator=torch.Generator().manual_seed(0))
            assert x.shape == (2, 1, 28, 28) and torch.isfinite(x).all()
            # second-order samplers can only hit odd budgets (2N - 1); they never exceed the budget by more than 1
            assert abs(net.calls - nfe) <= 1, (sampler, nfe, net.calls)
            if sampler not in method.second_order:
                assert net.calls == nfe


def test_ddpm_schedules():
    for schedule in ("linear", "cosine"):
        a = DDPM(1000, schedule).abar
        assert a.shape == (1000,)
        assert (a[1:] < a[:-1]).all(), "abar must decrease"
        assert a[0] > 0.99 and a[-1] < 1e-3


@pytest.mark.parametrize("pred", ["eps", "x0", "v"])
def test_ddpm_parameterisations_are_equivalent(pred):
    """Given the *true* target, to_x0_eps must recover the true x0 and eps exactly."""
    m = DDPM(1000, "cosine", pred, clip=False)
    x0, eps = torch.rand(2, 1, 4, 4) * 2 - 1, torch.randn(2, 1, 4, 4)
    t = torch.tensor([10, 700])
    a = m.abar[t].reshape(-1, 1, 1, 1)
    xt = m.q_sample(x0, t, eps)
    target = {"eps": eps, "x0": x0, "v": a.sqrt() * eps - (1 - a).sqrt() * x0}[pred]
    x0_hat, eps_hat = m.to_x0_eps(target, xt, a)
    assert torch.allclose(x0_hat, x0, atol=1e-4) and torch.allclose(eps_hat, eps, atol=1e-4)


def test_score_sde_marginals():
    t = torch.linspace(0, 1, 11)
    m, s = ScoreSDE("vp").marginal(t)
    assert torch.allclose(m ** 2 + s ** 2, torch.ones_like(t), atol=1e-6)  # variance preserving
    m, s = ScoreSDE("ve", sigma_min=0.01, sigma_max=50).marginal(torch.tensor([0.0, 1.0]))
    assert torch.allclose(s, torch.tensor([0.01, 50.0])) and (m == 1).all()


def test_ve_g2_is_derivative_of_variance():
    sde, t, h = ScoreSDE("ve"), torch.tensor([0.4]), 1e-4
    ds2 = (sde.marginal(t + h)[1] ** 2 - sde.marginal(t - h)[1] ** 2) / (2 * h)
    assert torch.allclose(sde.drift_g2(torch.zeros(1, 1, 1, 1), t)[1], ds2, rtol=1e-3)


def test_edm_sigmas_and_preconditioning():
    e = EDM()
    s = e.sigmas(18)
    assert math.isclose(s[0].item(), 80, rel_tol=1e-5) and math.isclose(s[-2].item(), 0.002, rel_tol=1e-4)
    assert s[-1] == 0 and (s[1:] < s[:-1]).all()
    sig = torch.tensor([0.01, 0.5, 50.0])
    c_skip, _c_out, c_in, _ = e.precond(sig)
    # for x = x0 + sigma n with Var(x0) = sd^2, the network input c_in x has unit variance
    assert torch.allclose(c_in ** 2 * (sig ** 2 + 0.25), torch.ones(3))
    assert c_skip[0] > 0.99 and c_skip[-1] < 1e-3  # small noise: keep x; large noise: predict from scratch


class Oracle(torch.nn.Module):
    """The exact optimal network when the data is a single image x*: tests the samplers, not the training."""

    def __init__(self, method, target):
        super().__init__()
        self.method, self.target, self.num_classes = method, target, 0
        self.in_channels, self.image_size = target.shape[0], target.shape[1]
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, t, y=None):
        m, xs = self.method, self.target.expand_as(x)
        b = lambda v: v.reshape(-1, 1, 1, 1)
        if isinstance(m, DDPM):
            a = b(m.abar[t.round().long()])
            eps = (x - a.sqrt() * xs) / (1 - a).sqrt()
            return {"eps": eps, "x0": xs, "v": a.sqrt() * eps - (1 - a).sqrt() * xs}[m.prediction]
        if isinstance(m, ScoreSDE):
            mm, s = m.marginal(t / 999)
            x = x * b((mm ** 2 + s ** 2).sqrt())  # undo c_in
            return (x - b(mm) * xs) / b(s)
        if isinstance(m, FlowMatching):
            return (x - xs) / b(t / 999)
        sigma = (t / 250).exp()  # EDM: invert c_noise
        c_skip, c_out, c_in, _ = m.precond(sigma)
        return (xs - b(c_skip) * x / b(c_in)) / b(c_out)


@pytest.mark.parametrize("mc", ALL, ids=method_label)
def test_samplers_reach_the_data_with_an_exact_denoiser(mc):
    target = torch.linspace(-0.8, 0.8, 16).reshape(1, 4, 4)
    method = build_method(mc)
    for sampler in method.samplers:
        x = generate(Oracle(method, target), method, 3, 30, sampler)
        err = (x - target).abs().max().item()
        assert err < 0.05, f"{sampler}: max error {err:.3f}"


def test_guidance_extrapolates(small_cfg):
    net = build_backbone(small_cfg("unet", 3))
    for p in net.parameters():  # make the network non-trivial
        torch.nn.init.normal_(p, std=0.05)
    x, t, y = torch.randn(2, 1, 28, 28), torch.rand(2) * 999, torch.tensor([0, 2])
    cond, uncond = net(x, t, y), net(x, t, torch.tensor([3, 3]))
    assert torch.allclose(guided(net, x, t, y, 1.0), cond)
    assert torch.allclose(guided(net, x, t, y, 0.0), uncond, atol=1e-6)
    assert torch.allclose(guided(net, x, t, y, 3.0), uncond + 3 * (cond - uncond), atol=1e-5)


def test_unknown_names_raise():
    with pytest.raises(ValueError):
        build_method(MethodConfig("nope"))
    with pytest.raises(ValueError):
        DDPM(prediction="score")
    with pytest.raises(ValueError):
        DDPM().sample(None, 1, 1, "euler")
