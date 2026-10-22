#!/usr/bin/env python3
"""
Direct ADC-vs-calibrated-charge comparison: for each task/metric, the
mean pretrained-vs-random-init delta (+/- std across seeds) under each E
feature, side by side. The figure the whole E-feature investigation was
building toward.

Usage:
    python plot_efeature_comparison.py --out efeature_comparison.png
"""
import argparse
import glob
import json
import os

import numpy as np


def load_deltas(pattern, metric_key):
    deltas = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            d = json.load(f)
        deltas.append(d[metric_key]["pretrained_mean"] - d[metric_key]["random_init_mean"])
    return np.array(deltas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="efeature_comparison.png")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("tracking ARI",
         load_deltas("results/tracking_multiseed_12k_adc_bonus_seeds/seed*/summary.json", "ari"),
         load_deltas("results/tracking_multiseed_12k_qcharge/seed*/summary.json", "ari")),
        ("tracking frac_found",
         load_deltas("results/tracking_multiseed_12k_adc_bonus_seeds/seed*/summary.json", "frac_found"),
         load_deltas("results/tracking_multiseed_12k_qcharge/seed*/summary.json", "frac_found")),
        ("PID macro-F1",
         load_deltas("results/pid_multiseed_12k_adc_bonus_seeds/seed*/summary.json", "macro_f1"),
         load_deltas("results/pid_multiseed_12k_qcharge/seed*/summary.json", "macro_f1")),
        ("PID balanced accuracy",
         load_deltas("results/pid_multiseed_12k_adc_bonus_seeds/seed*/summary.json", "balanced_accuracy"),
         load_deltas("results/pid_multiseed_12k_qcharge/seed*/summary.json", "balanced_accuracy")),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(panels))
    w = 0.32
    adc_means = [p[1].mean() for p in panels]; adc_stds = [p[1].std() for p in panels]
    q_means = [p[2].mean() for p in panels]; q_stds = [p[2].std() for p in panels]

    ax.bar(x - w/2, adc_means, w, yerr=adc_stds, capsize=4, color="tab:blue",
           label="TPCHits_adc (n=10 tracking / n=12 PID seeds)")
    ax.bar(x + w/2, q_means, w, yerr=q_stds, capsize=4, color="tab:orange",
           label="TPCHits_q calibrated charge (same n per task)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([p[0] for p in panels], rotation=15, ha="right")
    ax.set_ylabel("mean pretrained-vs-random-init delta\n(error bars = std across seeds)")
    ax.set_title("E-feature comparison: pretrained-vs-random-init effect size, ADC vs. calibrated charge\n"
                 "(both verified-different datasets, both multi-seed, same 12,000-event scale)")
    ax.legend()
    for i, (name, adc, q) in enumerate(panels):
        ax.text(i - w/2, adc_means[i] + adc_stds[i] + 0.002, f"{adc_means[i]:+.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, q_means[i] + q_stds[i] + 0.002, f"{q_means[i]:+.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"[DONE] Saved {args.out}")
    for name, adc, q in panels:
        print(f"{name}: ADC {adc.mean():+.4f}+/-{adc.std():.4f} (n={len(adc)}), "
              f"charge {q.mean():+.4f}+/-{q.std():.4f} (n={len(q)})")


if __name__ == "__main__":
    main()
