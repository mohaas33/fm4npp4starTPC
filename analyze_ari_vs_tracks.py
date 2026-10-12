#!/usr/bin/env python3
"""
Plots ARI (and frac_found) vs. true track count per event, for both arms,
using the test_results.pkl files eval_track_finding_star.py's --results-dir
already saved -- no retraining or GPU needed, reads straight off disk.

Usage:
    python analyze_ari_vs_tracks.py --results-dir results/tracking_pythia_multifile \
        --out ari_vs_tracks.png
"""

import argparse
import os
import pickle

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out", default="ari_vs_tracks.png")
    p.add_argument("--bin-width", type=int, default=5, help="Track-count bin width for the trend line.")
    return p.parse_args()


def load_arm(results_dir, arm_label):
    path = os.path.join(os.path.expanduser(results_dir), arm_label, "test_results.pkl")
    with open(path, "rb") as f:
        recs = pickle.load(f)
    n_tracks = np.array([len(np.unique(r["true_labels"])) for r in recs])
    ari = np.array([r["ari"] for r in recs])
    frac_found = np.array([r["frac_found"] for r in recs])
    return n_tracks, ari, frac_found


def binned_trend(n_tracks, values, bin_width):
    max_t = n_tracks.max()
    edges = np.arange(1, max_t + bin_width + 1, bin_width)
    centers, means, stds, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (n_tracks >= lo) & (n_tracks < hi)
        if m.sum() == 0:
            continue
        centers.append((lo + hi - 1) / 2)
        means.append(values[m].mean())
        stds.append(values[m].std())
        counts.append(int(m.sum()))
    return np.array(centers), np.array(means), np.array(stds), counts


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"pretrained": "tab:blue", "random_init": "tab:orange"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for arm_label in ("pretrained", "random_init"):
        n_tracks, ari, frac_found = load_arm(args.results_dir, arm_label)
        c = colors[arm_label]

        ax = axes[0]
        ax.scatter(n_tracks, ari, s=14, alpha=0.25, color=c, label=None)
        centers, means, stds, counts = binned_trend(n_tracks, ari, args.bin_width)
        ax.plot(centers, means, color=c, linewidth=2, marker="o",
                 label=f"{arm_label} (binned mean)")
        ax.fill_between(centers, means - stds, means + stds, color=c, alpha=0.15)

        ax = axes[1]
        ax.scatter(n_tracks, frac_found, s=14, alpha=0.25, color=c, label=None)
        centers2, means2, stds2, counts2 = binned_trend(n_tracks, frac_found, args.bin_width)
        ax.plot(centers2, means2, color=c, linewidth=2, marker="o",
                 label=f"{arm_label} (binned mean)")
        ax.fill_between(centers2, means2 - stds2, means2 + stds2, color=c, alpha=0.15)

    axes[0].set_xlabel("number of true tracks in event"); axes[0].set_ylabel("ARI")
    axes[0].set_title("ARI vs. event complexity"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("number of true tracks in event"); axes[1].set_ylabel("frac_found (double-majority)")
    axes[1].set_title("frac_found vs. event complexity"); axes[1].legend(fontsize=8)
    fig.suptitle(f"{os.path.basename(os.path.normpath(args.results_dir))}: "
                 f"points = individual test events, lines = binned mean +/- std (bin width={args.bin_width})")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=140)
    print(f"[DONE] Saved {args.out}")
    for arm_label in ("pretrained", "random_init"):
        n_tracks, ari, frac_found = load_arm(args.results_dir, arm_label)
        print(f"{arm_label}: n={len(ari)} events, track-count range {n_tracks.min()}-{n_tracks.max()}, "
              f"overall ARI {ari.mean():.3f} +/- {ari.std():.3f}")


if __name__ == "__main__":
    main()
