"""Every diffusion / flow formulation, each in one class with the same two methods:

    loss(net, x0, y)                     -> scalar training loss for a batch of clean images x0
    sample(net, n, steps, sampler, ...)  -> n generated images in [-1, 1] (roughly)

They differ only in three choices, which is the whole point of comparing them:

1. **how noise is added** (the forward process: which x_t for which t),
2. **what the network predicts** (noise eps, clean image x0, v, velocity, or a preconditioned D),
3. **how the reverse process is integrated** (ancestral / DDIM steps, an SDE or an ODE solver).

| class         | forward process                     | network target              | samplers               |
|---------------|-------------------------------------|-----------------------------|------------------------|
| DDPM          | x_t = sqrt(a_t) x0 + sqrt(1-a_t) e  | e, x0 or v                  | ddpm (ancestral), ddim |
| ScoreSDE (VP) | same, continuous t, beta(t) linear  | e  (score = -e / sigma)     | em (reverse SDE), ode  |
| ScoreSDE (VE) | x_t = x0 + sigma(t) e               | e  (score = -e / sigma)     | em, ode                |
| FlowMatching  | x_t = (1-t) x0 + t e                | velocity e - x0             | euler, heun            |
| EDM           | x = x0 + sigma e, sigma log-normal  | preconditioned denoiser D   | heun, euler            |

Classifier-free guidance works the same way for all of them: `guided()` runs the network with the
real label and with the null label and extrapolates, out = uncond + w (cond - uncond).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from diffusion.config import MethodConfig


def guided(net: nn.Module, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None, w: float) -> torch.Tensor:
    """Network output with classifier-free guidance (Ho & Salimans 2022).

    w = 1 is the plain conditional model, w = 0 the unconditional one, w > 1 pushes samples
    towards what makes the label more likely. One call costs two network evaluations when w != 1,
    batched together.
    """
    if y is None or w == 1.0:
        return net(x, t, y)
    null = torch.full_like(y, net.num_classes)
    out = net(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null]))
    cond, uncond = out.chunk(2)
    return uncond + w * (cond - uncond)


def _bcast(v: torch.Tensor) -> torch.Tensor:
    """(B,) -> (B, 1, 1, 1) so a per-image scalar multiplies a whole image."""
    return v.reshape(-1, 1, 1, 1)


class Method:
    """Interface shared by all formulations (see the module docstring)."""

    name = "base"
    samplers: tuple[str, ...] = ()
    second_order: tuple[str, ...] = ()  # samplers that call the network twice per step

    def loss(self, net, x0, y=None):
        raise NotImplementedError

    def sample(self, net, n, steps, sampler=None, y=None, guidance=1.0, shape=None, generator=None, record=None):
        raise NotImplementedError

    def steps_for_nfe(self, sampler: str, nfe: int) -> int:
        """Number of solver steps that costs `nfe` network evaluations (Heun = 2 per step, last one 1)."""
        return max(1, (nfe + 1) // 2) if sampler in self.second_order else nfe

    def _noise(self, shape, generator, device):
        return torch.randn(shape, generator=generator, device="cpu").to(device)


# ---------------------------------------------------------------------------------------------- DDPM
class DDPM(Method):
    """Denoising Diffusion Probabilistic Models (Ho et al. 2020), discrete T steps.

    Forward: q(x_t | x0) = N(sqrt(abar_t) x0, (1 - abar_t) I), abar_t = prod_{s<=t} (1 - beta_s).
    Loss (the simplified ELBO): || net(x_t, t) - target ||^2 with target eps, x0, or
    v = sqrt(abar) eps - sqrt(1 - abar) x0 (Salimans & Ho 2022). All three carry the same information:
    `to_x0_eps` converts any of them back to (x0_hat, eps_hat), and the sampler only uses those two.
    Sampling: the generalised DDIM update (Song et al. 2021) with eta = 1 is DDPM's ancestral
    sampler, eta = 0 is the deterministic DDIM sampler; both work on a strided subset of the T steps.
    """

    name = "ddpm"
    samplers = ("ddim", "ddpm")

    def __init__(self, timesteps: int = 1000, schedule: str = "linear", prediction: str = "eps", clip: bool = True):
        if prediction not in ("eps", "x0", "v"):
            raise ValueError(f"prediction must be eps | x0 | v, got {prediction!r}")
        self.T, self.prediction, self.clip = timesteps, prediction, clip
        if schedule == "linear":   # Ho et al.: beta from 1e-4 to 0.02 (scaled for other T)
            scale = 1000 / timesteps
            betas = torch.linspace(scale * 1e-4, scale * 0.02, timesteps, dtype=torch.float64)
            abar = torch.cumprod(1 - betas, 0)
        elif schedule == "cosine":  # Nichol & Dhariwal 2021: abar follows a squared cosine
            s = 0.008
            steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
            f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
            betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
            abar = torch.cumprod(1 - betas, 0)
        else:
            raise ValueError(f"schedule must be linear | cosine, got {schedule!r}")
        self.abar = abar.float()

    def _abar(self, t: torch.Tensor) -> torch.Tensor:
        return self.abar.to(t.device)[t]

    def q_sample(self, x0, t, eps):
        a = _bcast(self._abar(t))
        return a.sqrt() * x0 + (1 - a).sqrt() * eps

    def loss(self, net, x0, y=None):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        eps = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps)
        a = _bcast(self._abar(t))
        target = {"eps": eps, "x0": x0, "v": a.sqrt() * eps - (1 - a).sqrt() * x0}[self.prediction]
        return F.mse_loss(net(xt, t.float(), y), target)

    def to_x0_eps(self, out, xt, a):
        """Any parameterisation -> (x0_hat, eps_hat); `a` is abar_t broadcast to the image shape."""
        sa, s1a = a.sqrt(), (1 - a).sqrt()
        if self.prediction == "eps":
            eps = out
            x0 = (xt - s1a * eps) / sa
        elif self.prediction == "x0":
            x0 = out
            eps = (xt - sa * x0) / s1a
        else:  # v
            x0 = sa * xt - s1a * out
            eps = s1a * xt + sa * out
        if self.clip:  # the data lives in [-1, 1]: clip x0_hat and make eps consistent with it
            x0 = x0.clamp(-1, 1)
            eps = (xt - sa * x0) / s1a
        return x0, eps

    @torch.no_grad()
    def sample(self, net, n, steps, sampler=None, y=None, guidance=1.0, shape=None, generator=None, record=None):
        sampler = sampler or "ddim"
        if sampler not in self.samplers:
            raise ValueError(f"ddpm samplers: {self.samplers}, got {sampler!r}")
        eta = 1.0 if sampler == "ddpm" else 0.0
        device = next(net.parameters()).device
        ts = torch.linspace(self.T - 1, 0, min(steps, self.T)).round().long().tolist()
        x = self._noise((n, *shape), generator, device)
        for i, t in enumerate(ts):
            a = self.abar[t].item()
            a_prev = self.abar[ts[i + 1]].item() if i + 1 < len(ts) else 1.0
            tt = torch.full((n,), float(t), device=device)
            x0, eps = self.to_x0_eps(guided(net, x, tt, y, guidance), x, torch.tensor(a, device=device))
            # posterior std of the generalised update; eta = 1 gives DDPM's beta_tilde
            sigma = eta * math.sqrt((1 - a_prev) / (1 - a) * (1 - a / a_prev))
            x = math.sqrt(a_prev) * x0 + math.sqrt(max(1 - a_prev - sigma ** 2, 0.0)) * eps
            if sigma > 0 and i + 1 < len(ts):
                x = x + sigma * self._noise(x.shape, generator, device)
            if record is not None:
                record.append((t / self.T, x.cpu(), x0.cpu()))
        return x


# ----------------------------------------------------------------------------------------- Score SDE
class ScoreSDE(Method):
    """Score-based generative modelling with SDEs (Song et al. 2021), t continuous in [eps, 1].

    Forward SDE dx = f(x, t) dt + g(t) dw, whose marginal is x_t = m(t) x0 + s(t) e:
      VP (variance preserving, the continuous DDPM): f = -beta(t) x / 2, g^2 = beta(t), beta linear 0.1 -> 20
      VE (variance exploding, NCSN):                  f = 0, g^2 = d s^2/dt, s(t) = s_min (s_max/s_min)^t
    The network predicts e (denoising score matching): score(x, t) = grad log p_t(x) = -e_hat / s(t).
    Reverse-time SDE (sampler `em`, Euler-Maruyama): dx = [f - g^2 score] dt + g dw_bar, from t = 1 to eps.
    Probability-flow ODE (sampler `ode`, Heun):     dx = [f - g^2 score / 2] dt, same marginals, no noise.
    Both samplers end with one denoising step, x0_hat = (x - s e_hat) / m.
    """

    name = "score-sde"
    samplers = ("em", "ode")
    second_order = ("ode",)

    def __init__(self, kind: str = "vp", beta_min=0.1, beta_max=20.0, sigma_min=0.01, sigma_max=50.0, t_eps=1e-3):
        if kind not in ("vp", "ve"):
            raise ValueError(f"sde must be vp | ve, got {kind!r}")
        self.kind, self.t_eps = kind, t_eps
        self.beta_min, self.beta_max = beta_min, beta_max
        self.sigma_min, self.sigma_max = sigma_min, sigma_max

    def marginal(self, t: torch.Tensor):
        """(m(t), s(t)) with x_t = m x0 + s e."""
        if self.kind == "vp":
            log_m = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
            return log_m.exp(), (-torch.expm1(2 * log_m)).sqrt()
        return torch.ones_like(t), self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def drift_g2(self, x, t):
        """(f(x, t), g(t)^2) of the forward SDE."""
        if self.kind == "vp":
            beta = self.beta_min + t * (self.beta_max - self.beta_min)
            return -0.5 * _bcast(beta) * x, beta
        s = self.marginal(t)[1]
        return torch.zeros_like(x), 2 * s ** 2 * math.log(self.sigma_max / self.sigma_min)

    def eps_hat(self, net, x, t, y=None, w=1.0):
        m, s = self.marginal(t)
        c_in = _bcast(1 / (m ** 2 + s ** 2).sqrt())  # keeps the VE input at unit scale (s goes up to 50)
        return guided(net, c_in * x, t * 999, y, w)

    def loss(self, net, x0, y=None):
        t = self.t_eps + (1 - self.t_eps) * torch.rand(x0.shape[0], device=x0.device)
        e = torch.randn_like(x0)
        m, s = self.marginal(t)
        xt = _bcast(m) * x0 + _bcast(s) * e
        return F.mse_loss(self.eps_hat(net, xt, t, y), e)

    def _velocity(self, net, x, t, y, w, ode: bool):
        """Drift of the reverse SDE (ode=False) or of the probability-flow ODE (ode=True), and x0_hat."""
        eps = self.eps_hat(net, x, t, y, w)
        m, s = self.marginal(t)
        score = -eps / _bcast(s)
        f, g2 = self.drift_g2(x, t)
        g2 = _bcast(g2)
        x0 = (x - _bcast(s) * eps) / _bcast(m)
        return f - (0.5 if ode else 1.0) * g2 * score, g2, x0

    @torch.no_grad()
    def sample(self, net, n, steps, sampler=None, y=None, guidance=1.0, shape=None, generator=None, record=None):
        sampler = sampler or "em"
        if sampler not in self.samplers:
            raise ValueError(f"score-sde samplers: {self.samplers}, got {sampler!r}")
        device = next(net.parameters()).device
        ts = torch.linspace(1.0, self.t_eps, steps + 1, device=device)
        prior_std = self.marginal(torch.ones(1))[1].item()  # ~1 for VP, sigma_max for VE
        x = prior_std * self._noise((n, *shape), generator, device)
        for i in range(steps):
            t, t_next = ts[i].expand(n), ts[i + 1].expand(n)
            dt = (ts[i + 1] - ts[i]).item()  # negative: we integrate backwards in time
            drift, g2, x0 = self._velocity(net, x, t, y, guidance, ode=(sampler == "ode"))
            if i + 1 == steps:
                # last step: jump to the denoised estimate E[x0 | x_t] (Tweedie), Song et al.'s "noise removal".
                # A plain Euler step here overshoots: near t = 0 the step size exceeds s(t)^2 / g(t)^2.
                x = x0
            elif sampler == "em":  # Euler-Maruyama on the reverse SDE: deterministic drift + fresh noise
                x = x + drift * dt + (g2 * -dt).sqrt() * self._noise(x.shape, generator, device)
            else:  # Heun on the probability-flow ODE
                x_next = x + drift * dt
                d2, _, _ = self._velocity(net, x_next, t_next, y, guidance, ode=True)
                x = x + 0.5 * (drift + d2) * dt
            if record is not None:
                record.append((ts[i + 1].item(), x.cpu(), x0.cpu()))
        return x


# ------------------------------------------------------------------------------------- Flow matching
class FlowMatching(Method):
    """Flow matching / rectified flow (Lipman et al. 2023, Liu et al. 2023), t in [0, 1].

    Straight-line interpolant x_t = (1 - t) x0 + t e between data (t = 0) and noise (t = 1).
    Its time derivative is the constant velocity v = e - x0, and the network regresses it:
    || net(x_t, t) - (e - x0) ||^2. Sampling integrates the ODE dx/dt = v(x, t) from t = 1 down to
    t = 0 with Euler or Heun. No noise schedule, no SDE: the simplest formulation in this repo.
    """

    name = "flow-matching"
    samplers = ("euler", "heun")
    second_order = ("heun",)

    def loss(self, net, x0, y=None):
        t = torch.rand(x0.shape[0], device=x0.device)
        e = torch.randn_like(x0)
        xt = (1 - _bcast(t)) * x0 + _bcast(t) * e
        return F.mse_loss(net(xt, t * 999, y), e - x0)

    @torch.no_grad()
    def sample(self, net, n, steps, sampler=None, y=None, guidance=1.0, shape=None, generator=None, record=None):
        sampler = sampler or "euler"
        if sampler not in self.samplers:
            raise ValueError(f"flow-matching samplers: {self.samplers}, got {sampler!r}")
        device = next(net.parameters()).device
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        x = self._noise((n, *shape), generator, device)
        for i in range(steps):
            t, t_next = ts[i].expand(n), ts[i + 1].expand(n)
            dt = (ts[i + 1] - ts[i]).item()
            v = guided(net, x, t * 999, y, guidance)
            x0 = x - _bcast(t) * v  # where the straight line through x with slope v lands at t = 0
            x_next = x + v * dt
            if sampler == "heun" and i + 1 < steps:
                v2 = guided(net, x_next, t_next * 999, y, guidance)
                x_next = x + 0.5 * (v + v2) * dt
            x = x_next
            if record is not None:
                record.append((ts[i + 1].item(), x.cpu(), x0.cpu()))
        return x


# ----------------------------------------------------------------------------------------------- EDM
class EDM(Method):
    """Elucidating the Design Space of Diffusion Models (Karras et al. 2022).

    x = x0 + sigma e. The network F is wrapped in a preconditioned denoiser
        D(x, sigma) = c_skip x + c_out F(c_in x, c_noise),
    with c_in = 1/sqrt(sigma^2 + sd^2), c_skip = sd^2/(sigma^2 + sd^2), c_out = sigma sd/sqrt(sigma^2 + sd^2),
    c_noise = ln(sigma)/4, sd = sigma_data = 0.5. The scalings keep F's input and target at unit
    variance at every noise level, so one network is easy to train across sigma = 0.002 ... 80.
    Training draws ln(sigma) ~ N(-1.2, 1.2^2). Sampling solves the ODE dx/dsigma = (x - D(x, sigma)) / sigma
    with Heun's method on the Karras schedule sigma_i = (smax^(1/rho) + i/(N-1) (smin^(1/rho) - smax^(1/rho)))^rho.
    """

    name = "edm"
    samplers = ("heun", "euler")
    second_order = ("heun",)

    def __init__(self, sigma_data=0.5, p_mean=-1.2, p_std=1.2, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        self.sd, self.p_mean, self.p_std = sigma_data, p_mean, p_std
        self.sigma_min, self.sigma_max, self.rho = sigma_min, sigma_max, rho

    def precond(self, sigma):
        sd2 = self.sd ** 2
        c_skip = sd2 / (sigma ** 2 + sd2)
        c_out = sigma * self.sd / (sigma ** 2 + sd2).sqrt()
        c_in = 1 / (sigma ** 2 + sd2).sqrt()
        c_noise = sigma.log() / 4
        return c_skip, c_out, c_in, c_noise

    def denoise(self, net, x, sigma, y=None, w=1.0):
        c_skip, c_out, c_in, c_noise = self.precond(sigma)
        f = guided(net, _bcast(c_in) * x, 1000 * c_noise, y, w)
        return _bcast(c_skip) * x + _bcast(c_out) * f

    def loss(self, net, x0, y=None):
        sigma = (torch.randn(x0.shape[0], device=x0.device) * self.p_std + self.p_mean).exp()
        x = x0 + _bcast(sigma) * torch.randn_like(x0)
        c_skip, c_out, c_in, c_noise = self.precond(sigma)
        # lambda(sigma) ||D - x0||^2 with lambda = 1/c_out^2 is exactly this unit-variance regression on F:
        target = (x0 - _bcast(c_skip) * x) / _bcast(c_out)
        return F.mse_loss(net(_bcast(c_in) * x, 1000 * c_noise, y), target)

    def sigmas(self, steps: int, device="cpu") -> torch.Tensor:
        i = torch.arange(steps, dtype=torch.float64, device=device)
        a, b = self.sigma_max ** (1 / self.rho), self.sigma_min ** (1 / self.rho)
        s = (a + i / max(steps - 1, 1) * (b - a)) ** self.rho
        return torch.cat([s, torch.zeros(1, dtype=s.dtype, device=device)]).float()

    @torch.no_grad()
    def sample(self, net, n, steps, sampler=None, y=None, guidance=1.0, shape=None, generator=None, record=None):
        sampler = sampler or "heun"
        if sampler not in self.samplers:
            raise ValueError(f"edm samplers: {self.samplers}, got {sampler!r}")
        device = next(net.parameters()).device
        sig = self.sigmas(steps, device)
        x = sig[0] * self._noise((n, *shape), generator, device)
        for i in range(steps):
            s, s_next = sig[i], sig[i + 1]
            x0 = self.denoise(net, x, s.expand(n), y, guidance)
            d = (x - x0) / s
            x_next = x + (s_next - s) * d
            if sampler == "heun" and s_next > 0:  # second-order correction, skipped on the last step
                d2 = (x_next - self.denoise(net, x_next, s_next.expand(n), y, guidance)) / s_next
                x_next = x + (s_next - s) * 0.5 * (d + d2)
            x = x_next
            if record is not None:
                record.append((s_next.item(), x.cpu(), x0.cpu()))
        return x


def build_method(cfg: MethodConfig) -> Method:
    if cfg.method == "ddpm":
        return DDPM(cfg.timesteps, cfg.schedule, cfg.prediction)
    if cfg.method == "score-sde":
        return ScoreSDE(cfg.sde)
    if cfg.method == "flow-matching":
        return FlowMatching()
    if cfg.method == "edm":
        return EDM()
    raise ValueError(f"unknown method {cfg.method!r} (ddpm | score-sde | flow-matching | edm)")


def method_label(cfg: MethodConfig) -> str:
    """Short human name used in tables, e.g. 'ddpm-v-cosine' or 'score-sde-ve'."""
    if cfg.method == "ddpm":
        return f"ddpm-{cfg.prediction}-{cfg.schedule}"
    if cfg.method == "score-sde":
        return f"score-sde-{cfg.sde}"
    return cfg.method
