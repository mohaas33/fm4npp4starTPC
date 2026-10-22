#!/usr/bin/env python3
"""
Same idea as plot_multiseed_distributions.py but for eval_trackfit_star.py's
summary.json format: RMS residuals (lower = better), three regression
targets (mcP, mcPt, mcEta) instead of two classification/clustering metrics.

Usage:
    python plot_trackfit_distributions.py --out trackfit_multiseed_distributions.png
"""
import argparse
import glob
import json
import os

import numpy as np

TARGET_NAMES = ["mcP", "mcPt", "mcEta"]


def load_seeds(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        seed = os.path.basename(os.path.dirname(path)).replace("seed", "")
        with open(path) as f:
            d = json.load(f)
        row = {"seed": seed}
        for name in TARGET_NAMES:
            row[name] = (d[name]["pretrained_mean"], d[name]["random_init_mean"])
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/trackfit_multiseed_12k")
    p.add_argument("--out", default="trackfit_multiseed_distributions.png")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_seeds(f"{args.results_dir}/seed*/summary.json")
    n = len(rows)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for col, name in enumerate(TARGET_NAMES):
        pre = np.array([r[name][0] for r in rows])
        rnd = np.array([r[name][1] for r in rows])
        deltas = rnd - pre  # positive = pretrained has LOWER (better) RMS

        ax = axes[0, col]
        for i in range(n):
            ax.plot([0, 1], [rnd[i], pre[i]], color="gray", alpha=0.6, linewidth=1, zorder=1)
        ax.scatter(np.zeros(n), rnd, color="tab:orange", s=60, zorder=2, label="random-init")
        ax.scatter(np.ones(n), pre, color="tab:blue", s=60, zorder=2, label="pretrained")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["random-init", "pretrained"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylabel(f"RMS({name}) [physical units]")
        ax.set_title(f"{name}: RMS residual (lower = better)\n(n={n} seeds, each line = one seed)")
        if col == 0:
            ax.legend(fontsize=8, loc="upper right")

        ax = axes[1, col]
        colors = ["tab:green" if d > 0 else "tab:red" for d in deltas]
        ax.bar(range(n), deltas, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(n)); ax.set_xticklabels([r["seed"] for r in rows], fontsize=8)
        ax.set_xlabel("seed")
        ax.set_ylabel("improvement (random-pretrained)")
        ax.set_title(f"mean={deltas.mean():+.4f}, all seeds improved={bool(np.all(deltas > 0))}", fontsize=9)

    fig.suptitle(f"Track-fitting (first attempt at this 3rd downstream task): "
                 f"per-hit kinematics regression, n={n} seeds, 12,000 train events/arm")
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=3.0)
    fig.savefig(args.out, dpi=140)
    print(f"[DONE] Saved {args.out}")


if __name__ == "__main__":
    main()
