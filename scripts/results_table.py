"""Gather runs/*/eval.json into assets/: results.json, results.md (README table) and figures.

    python scripts/results_table.py [--runs-dir runs] [--assets assets]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from diffusion.viz import plot_losses

ORDER = ["ddpm-eps", "ddpm-x0", "ddpm-v", "score-sde-vp", "score-sde-ve", "flow-matching", "edm",
         "flow-matching-cfg", "flow-matching-dit"]


def fmt_time(s: float) -> str:
    return f"{s / 60:.1f} min"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--assets", default="assets")
    a = p.parse_args()
    runs_dir, assets = Path(a.runs_dir), Path(a.assets)
    assets.mkdir(exist_ok=True)
    results = {}
    for name in ORDER:
        f = runs_dir / name / "eval.json"
        if f.exists():
            results[name] = json.loads(f.read_text())
    if not results:
        raise SystemExit("no runs/*/eval.json yet")
    (assets / "results.json").write_text(json.dumps(results, indent=1))

    floor = sum(r["floor"] for r in results.values()) / len(results)
    nfes = (5, 10, 20, 50, 100)
    lines = [
        "| run | backbone | params | train (500 steps) | " + " | ".join(f"NFE {n}" for n in nfes)
        + " | 2nd sampler NFE 10 / 50 | s / 100 img (NFE 20) |",
        "|---|---|---|---|" + "---|" * len(nfes) + "---|---|",
    ]
    curves = {}
    for name, r in results.items():
        prim = {row["nfe"]: row for row in r["rows"] if row["sampler"] == r["primary"] and row["guidance"] == 1.0}
        sec = {row["nfe"]: row for row in r["rows"] if row["sampler"] == r["secondary"] and row["guidance"] == 1.0}
        # second-order samplers land on odd NFE (2N - 1): match each target to the closest measured NFE
        def at(table, n):
            if not table:
                return None
            k = min(table, key=lambda m: abs(m - n))
            return table[k] if abs(k - n) <= 1 else None
        cells = []
        for n in nfes:
            row = at(prim, n)
            cells.append(f"{row['fid']:.2f}" if row else "–")
        s10, s50 = at(sec, 10), at(sec, 50)
        sec_cell = f"`{r['secondary']}` {s10['fid']:.2f} / {s50['fid']:.2f}" if s10 and s50 else "–"
        r20 = at(prim, 20)
        lines.append(f"| `{name}` (`{r['primary']}`) | {r['backbone']} | {r['params'] / 1e6:.2f} M | "
                     f"{fmt_time(r['train_seconds'])} | " + " | ".join(cells)
                     + f" | {sec_cell} | {r20['sec_per_100']:.1f} s |")
        curves[f"{name} ({r['primary']})"] = [(row["nfe"], row["fid"]) for row in prim.values()]
        if sec:
            curves[f"{name} ({r['secondary']})"] = [(row["nfe"], row["fid"]) for row in sec.values()]
    lines.append(f"\nreal vs real floor (200 training images vs the test set): {floor:.2f}")
    if "flow-matching-cfg" in results:
        r = results["flow-matching-cfg"]
        lines += ["", "| guidance w | NFE | FID | label accuracy |", "|---|---|---|---|"]
        for row in sorted((x for x in r["rows"] if x["nfe"] in (19, 20, 21) or x["nfe"] == 20),
                          key=lambda x: x["guidance"]):
            lines.append(f"| {row['guidance']:g} | {row['nfe']} | {row['fid']:.2f} | {row['label_acc']:.1%} |")
    table = "\n".join(lines)
    (assets / "results.md").write_text(table + "\n")
    print(table)

    plot_runs(results, assets / "nfe_curves.png", floor)
    hists = {n: json.loads((runs_dir / n / "history.json").read_text()) for n in results}
    plot_losses(hists, assets / "loss_curves.png", title="training loss, relative to the first 50 steps")
    for name in results:
        for f in ("samples.png", "trajectory.png", "samples_w1.png", "samples_w3.png"):
            src = runs_dir / name / f
            if src.exists():
                shutil.copy(src, assets / f"{name}_{f}")
    montage(results, runs_dir, assets / "samples_all.png")


def plot_runs(results: dict, path: Path, floor: float) -> None:
    """One colour per run: solid line = main sampler, dashed = second sampler."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = plt.get_cmap("tab10").colors
    for i, (name, r) in enumerate(results.items()):
        for sampler, ls in ((r["primary"], "-"), (r["secondary"], "--")):
            pts = sorted((row["nfe"], row["fid"]) for row in r["rows"]
                         if row["sampler"] == sampler and row["guidance"] == 1.0)
            if pts:
                ax.plot(*zip(*pts), ls=ls, marker="o", ms=3.5, lw=1.5, color=colors[i % 10],
                        label=f"{name} ({sampler})")
    ax.axhline(floor, color="gray", ls=":", lw=1, label="real vs real (floor)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("network evaluations per sample (NFE)")
    ax.set_ylabel("classifier-feature FID (lower is better)")
    ax.set_title("MNIST, 500 training steps, 200 samples per point")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def montage(results: dict, runs_dir: Path, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    names = list(results)
    cols = 3
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, name in zip(axes.flat, names):
        ax.imshow(mpimg.imread(runs_dir / name / "samples.png"))
        ax.set_title(f"{name} ({results[name]['primary']}, NFE ~50)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
