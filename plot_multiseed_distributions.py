#!/usr/bin/env python3
"""
Reads per-seed summary.json files from the overnight multi-seed runs
(results/tracking_multiseed_12k/seed*/summary.json,
results/pid_multiseed_12k/seed*/summary.json) and plots, per task/metric,
a paired per-seed slope plot (pretrained vs random-init, one line per seed)
plus the distribution of per-seed deltas -- the clearest way to show "all
seeds, same direction" at small n (5-6 seeds), where a standard boxplot is
not very informative.

Usage:
    python plot_multiseed_distributions.py --out multiseed_distributions.png
"""
import argparse
import glob
import json
import os

import numpy as np


def load_seeds(pattern, metric_keys):
    rows = []
    for path in sorted(glob.glob(pattern)):
        seed = os.path.basename(os.path.dirname(path)).replace("seed", "")
        with open(path) as f:
            d = json.load(f)
        row = {"seed": seed}
        for mk in metric_keys:
            row[mk] = (d[mk]["pretrained_mean"], d[mk]["random_init_mean"])
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="multiseed_distributions.png")
    p.add_argument("--tracking-dir", default="results/tracking_multiseed_12k")
    p.add_argument("--pid-dir", default="results/pid_multiseed_12k")
    p.add_argument("--title", default="Overnight multi-seed run (12,000 train events/arm)")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trk = load_seeds(f"{args.tracking_dir}/seed*/summary.json", ["ari", "frac_found"])
    pid = load_seeds(f"{args.pid_dir}/seed*/summary.json", ["macro_f1", "balanced_accuracy"])

    panels = [
        ("tracking", trk, "ari", "ARI"),
        ("tracking", trk, "frac_found", "frac_found"),
        ("PID", pid, "macro_f1", "macro-F1"),
        ("PID", pid, "balanced_accuracy", "balanced accuracy"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, (task, rows, key, label) in enumerate(panels):
        pre = np.array([r[key][0] for r in rows])
        rnd = np.array([r[key][1] for r in rows])
        deltas = pre - rnd
        n = len(rows)

        ax = axes[0, col]
        for i in range(n):
            ax.plot([0, 1], [rnd[i], pre[i]], color="gray", alpha=0.6, linewidth=1, zorder=1)
        ax.scatter(np.zeros(n), rnd, color="tab:orange", s=60, zorder=2, label="random-init")
        ax.scatter(np.ones(n), pre, color="tab:blue", s=60, zorder=2, label="pretrained")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["random-init", "pretrained"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylabel(label)
        ax.set_title(f"{task}: {label}\n(n={n} seeds, each line = one seed)")
        if col == 0:
            ax.legend(fontsize=8, loc="lower right")

        ax = axes[1, col]
        colors = ["tab:green" if d > 0 else "tab:red" for d in deltas]
        ax.bar(range(n), deltas, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(n)); ax.set_xticklabels([r["seed"] for r in rows], fontsize=8)
        ax.set_xlabel("seed")
        ax.set_ylabel(f"delta ({label})")
        ax.set_title(f"per-seed delta: mean={deltas.mean():+.4f}, "
                      f"all positive={bool(np.all(deltas > 0))}")

    fig.suptitle(f"{args.title}: tracking (n={len(trk)} seeds, 15 epochs) + PID (n={len(pid)} seeds, 20 epochs)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=140)
    print(f"[DONE] Saved {args.out}")


if __name__ == "__main__":
    main()
