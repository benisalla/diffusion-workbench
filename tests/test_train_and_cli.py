import json

import torch

from diffusion import evaluate, sample, train
from diffusion.config import RunConfig
from diffusion.data import load_dataset
from diffusion.evaluate import Evaluator, train_classifier


def _cfg(tmp_path, method="flow-matching", **train_kw):
    cfg = RunConfig()
    cfg.method.method = method
    cfg.model.base_channels, cfg.model.channel_mults = 8, (1, 2)
    cfg.train.output_dir = str(tmp_path / "run")
    cfg.train.batch_size, cfg.train.log_every, cfg.train.warmup_steps = 16, 10, 10
    for k, v in train_kw.items():
        setattr(cfg.train, k, v)
    return cfg


def test_learns_a_tiny_problem(tmp_path):
    """Overfit 8 shape images: the loss must drop well below its starting value."""
    x, y, k = load_dataset("shapes", train=False, n=8)
    cfg = _cfg(tmp_path, "flow-matching", max_steps=80, lr=3e-3, batch_size=8)
    res = train.train(cfg, data=(x, y, k), verbose=False)
    loss = res["history"]["loss"]
    assert loss[-1] < 0.5 * loss[0], loss


def test_checkpoint_round_trip_and_conditioning(tmp_path):
    x, y, k = load_dataset("shapes", train=False, n=32)
    cfg = _cfg(tmp_path, "edm", max_steps=3, cond=True)
    res = train.train(cfg, data=(x, y, k), verbose=False)
    net, method, cfg2 = sample.load_checkpoint(res["path"])
    assert cfg2.model.num_classes == 4 and cfg2.method.method == "edm"
    a = sample.generate(net, method, 4, 3, labels=torch.arange(4), guidance=2.0, seed=1)
    b = sample.generate(res["ema"], res["method"], 4, 3, labels=torch.arange(4), guidance=2.0, seed=1)
    assert torch.allclose(a, b)
    hist = json.loads((tmp_path / "run" / "history.json").read_text())
    assert hist["steps"] == 3 and hist["params"] > 0


def test_cli_train_sample_eval(tmp_path):
    out = tmp_path / "cli"
    train.main(["--method", "ddpm", "--prediction", "v", "--schedule", "cosine", "--dataset", "shapes",
                "--n-train", "64", "--base-channels", "8", "--channel-mults", "1", "2", "--max-steps", "2",
                "--batch-size", "8", "--sample-steps", "2", "--output-dir", str(out)])
    assert (out / "last.pt").exists() and (out / "samples.png").exists() and (out / "loss.png").exists()
    sample.main([str(out / "last.pt"), "--n", "4", "--steps", "3", "--sampler", "ddpm", "--out", str(out / "s.png"),
                 "--trajectory", str(out / "traj.png")])
    assert (out / "s.png").exists() and (out / "traj.png").exists()
    # evaluation with a classifier trained in a few steps on the shapes data
    (x, y, k), (xt, yt, _) = load_dataset("shapes", n=400), load_dataset("shapes", train=False, n=200)
    clf, acc = train_classifier("shapes", steps=30, data=((x, y), (xt, yt), k), verbose=False)
    assert 0 <= acc <= 1
    ev = Evaluator("shapes", clf=clf, test=(xt, yt, k))
    rows = evaluate.evaluate_checkpoint(str(out / "last.pt"), ev, nfes=(2, 4), n_samples=20, verbose=False)
    assert [r["nfe"] for r in rows] == [2, 4, 2, 4] and all(r["fid"] >= 0 for r in rows)


def test_dit_backbone_trains(tmp_path):
    x, y, k = load_dataset("shapes", train=False, n=16)
    cfg = _cfg(tmp_path, "score-sde", max_steps=2)
    cfg.model.backbone, cfg.model.dit_dim, cfg.model.dit_depth, cfg.model.patch_size = "dit", 32, 1, 7
    res = train.train(cfg, data=(x, y, k), verbose=False)
    assert res["history"]["steps"] == 2
