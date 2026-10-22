#!/usr/bin/env python3
"""
STAR-adapted track-FITTING (truth momentum/eta regression) evaluation --
mirrors eval_pid_star.py's structure exactly (same frozen-backbone,
AttentionHead/MambaHead reuse, --results-dir/--events-out pattern), but
regresses continuous truth kinematics instead of classifying species.

Uses convert_star_pythia_multifile.py's real reg_target = (mcP, mcPt, mcEta,
mcVr, 0, 0, 0) per hit -- unused by anything until now (flagged as groundwork
in README_Oct12_2026.md section 2). Validity mask is pid_target != UNKNOWN,
since both are populated together from the same T<->TT truth join (verified
by reading convert_star_pythia_multifile.py: reg[k,:] and pid[k] are only
ever set inside the same "entry is not None" branch).

Regresses (mcP, mcPt, mcEta) -- the three best-populated, most physically
meaningful of the four reg_target columns; mcVr (truth vertex radius) is
left out of this first version, not because it's unavailable but to keep
the first real attempt at this task simple. Each target z-scored using
TRAIN-set statistics before regression (targets have very different natural
scales: mcP/mcPt ~0-6 GeV, mcEta ~-2..2), metrics reported back in physical
units.

Loss: masked smooth-L1 (Huber) per component, summed -- robust to outliers,
standard choice for regression heads, unlike the plain MSE this project's
own pretraining objective uses elsewhere.

Metric: plain RMS of (pred - true) per component, in physical units, no
iterative sigma-clipping yet (the LHC minGRU trajectory-fitting paper
discussed in README_091626.md uses iteratively-3-sigma-clipped RMS as the
real physics standard -- flagged as a possible improvement for a later,
more careful pass, not implemented in this first version).

Usage:
    python eval_trackfit_star.py \
        --data-dir data/pythia510_multifile_16k \
        --checkpoint checkpoints/pp_nerf_m3_k30.ckpt \
        --n-fit-events 60 --n-train-events 3000 --n-test-events 300 \
        --head-epochs 20 --seed 42 \
        --results-dir results/trackfit_test \
        --out trackfit_summary.png
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from extract_embeddings_pca import compute_star_stats, SPHENIX_STATS
from star_voxelizer import compute_star_final_bins, build_voxelizer, hierarchical_raster_scan_order
from extract_embeddings_linear_probe import event_polar, build_model
from eval_track_finding_star import get_frozen_feature_stack, save_feature_stack, load_feature_stack

FM4NPP_REPO = os.path.expanduser(os.environ.get("FM4NPP_REPO", "~/Work/STAR/PP_collision"))
sys.path.append(FM4NPP_REPO)
sys.path.append(os.path.join(FM4NPP_REPO, "train", "downstream"))
from model import AttentionHead, MambaHead

UNKNOWN_CLASS = 6  # must match convert_star_pythia_multifile.py -- marks hits with no valid reg_target either
TARGET_NAMES = ["mcP", "mcPt", "mcEta"]  # reg_target columns 0,1,2 (mcVr, col 3, left out of this first version)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--n-fit-events", type=int, default=60)
    p.add_argument("--n-train-events", type=int, default=3000)
    p.add_argument("--n-test-events", type=int, default=300)
    p.add_argument("--min-hits", type=int, default=15)
    p.add_argument("--norm", default="star", choices=["star", "sphenix"])
    p.add_argument("--norm-sample-events", type=int, default=200)
    p.add_argument("--n-bins", type=int, nargs=3, default=[8, 8, 6], metavar=("ETA", "PHI", "RADIUS"))
    p.add_argument("--n-radial-groups", type=int, default=2)
    p.add_argument("--dim-sweep-order", type=int, nargs=3, default=[2, 1, 0], metavar=("D0", "D1", "D2"))
    p.add_argument("--revert-order", type=int, nargs=3, default=[2, 1, 0], metavar=("D0", "D1", "D2"))

    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--d-state", type=int, default=32)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--klen", type=int, default=30)

    p.add_argument("--head-type", default="attention", choices=["attention", "mamba"])
    p.add_argument("--head-embed-dim", type=int, default=256)
    p.add_argument("--head-layers", type=int, default=1)
    p.add_argument("--head-dropout", type=float, default=0.1)
    p.add_argument("--head-epochs", type=int, default=20)
    p.add_argument("--head-lr", type=float, default=1e-4)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="trackfit_summary.png")
    p.add_argument("--feature-cache-dir", default=None)
    p.add_argument("--results-dir", default=None)
    return p.parse_args()


def build_head(args, device):
    Head = AttentionHead if args.head_type == "attention" else MambaHead
    kwargs = dict(input_dim=args.embed_dim, embed_dim=args.head_embed_dim, num_layers=args.head_layers,
                  num_feature_layers=args.num_layers, num_output_dim=len(TARGET_NAMES),
                  dropout=args.head_dropout)
    if args.head_type == "attention":
        kwargs["num_heads"] = 4
    else:
        kwargs.update(d_state=64, d_conv=4, expand=2)
    return Head(**kwargs).to(device)


def forward_event(down_model, feature_stack_cpu, device):
    feature = feature_stack_cpu.to(device).unsqueeze(1)
    n_hits = feature.shape[2]
    dummy_x = torch.zeros(1, n_hits, 4, device=device)
    padding_mask = torch.ones(1, n_hits, dtype=torch.bool, device=device)
    pred_dict = down_model(dummy_x, feature, padding_mask=padding_mask, pretrain=True)
    return pred_dict["pred_logits"].squeeze(0)  # (N, 3) raw (un-normalized-back) regression output


def masked_huber_loss(pred_norm, target_norm, valid, device):
    if valid.sum() == 0:
        return None
    return F.smooth_l1_loss(pred_norm[valid], target_norm[valid].to(device))


def train_head(down_model, feature_paths, reg_list, pid_list, target_mean, target_std, epochs, lr, device):
    opt = torch.optim.AdamW(down_model.parameters(), lr=lr, weight_decay=1e-4)
    t_mean = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    t_std = torch.as_tensor(target_std, dtype=torch.float32, device=device)
    down_model.train()
    for ep in range(epochs):
        total, n_seen = 0.0, 0
        order = np.random.permutation(len(feature_paths))
        for idx in order:
            feat_stack = load_feature_stack(feature_paths[idx])
            reg = torch.as_tensor(reg_list[idx][:, :3], dtype=torch.float32)
            pid = torch.as_tensor(pid_list[idx], dtype=torch.long)
            valid = pid != UNKNOWN_CLASS
            if valid.sum() == 0:
                continue
            target_norm = (reg - t_mean.cpu()) / t_std.cpu()
            pred_norm = forward_event(down_model, feat_stack, device)
            loss = masked_huber_loss(pred_norm, target_norm, valid, device)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n_seen += 1
        if n_seen and (ep % 2 == 0 or ep == epochs - 1):
            print(f"    [trackfit-head] epoch {ep}: mean masked Huber loss = {total / n_seen:.4f}")
    down_model.eval()
    return down_model


def score_event(down_model, feat_stack, reg, pid, target_mean, target_std, device):
    reg = np.asarray(reg)[:, :3]
    pid = np.asarray(pid)
    valid = pid != UNKNOWN_CLASS
    if valid.sum() == 0:
        return None
    with torch.no_grad():
        pred_norm = forward_event(down_model, feat_stack, device).cpu().numpy()
    pred_phys = pred_norm * target_std + target_mean  # undo z-score, back to physical units
    resid = pred_phys[valid] - reg[valid]  # (n_valid, 3)
    rms = np.sqrt((resid ** 2).mean(axis=0))
    bias = resid.mean(axis=0)
    return dict(rms=rms, bias=bias, n_valid=int(valid.sum()), n_hits=len(pid),
                pred_phys=pred_phys, valid=valid)


def save_arm_results(results_dir, arm_label, down_model, backbone_model, is_random_init, test_records):
    arm_dir = os.path.join(results_dir, arm_label)
    os.makedirs(arm_dir, exist_ok=True)
    torch.save(down_model.state_dict(), os.path.join(arm_dir, "down_model.pt"))
    if is_random_init:
        torch.save(backbone_model.state_dict(), os.path.join(arm_dir, "backbone.pt"))
    with open(os.path.join(arm_dir, "test_results.pkl"), "wb") as f:
        pickle.dump(test_records, f)
    print(f"[INFO] Saved {arm_label} track-fit results to {arm_dir}/")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")

    from mmap_ninja import RaggedMmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sstats

    if args.results_dir:
        os.makedirs(os.path.expanduser(args.results_dir), exist_ok=True)
        with open(os.path.join(os.path.expanduser(args.results_dir), "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    feat_path = os.path.join(os.path.expanduser(args.data_dir), f"features_{args.split}")
    reg_path = os.path.join(os.path.expanduser(args.data_dir), f"reg_target_{args.split}")
    pid_path = os.path.join(os.path.expanduser(args.data_dir), f"pid_target_{args.split}")
    features_mmap = RaggedMmap(feat_path)
    reg_mmap = RaggedMmap(reg_path)
    pid_mmap = RaggedMmap(pid_path)
    n_avail = len(features_mmap)
    print(f"[INFO] {n_avail} events available in split '{args.split}'")

    need = args.n_fit_events + args.n_train_events + args.n_test_events
    if need > n_avail:
        sys.exit(f"[ERROR] need {need} events but only {n_avail} available.")

    events_with_hits = []
    for i in range(n_avail):
        if np.asarray(features_mmap[i]).shape[0] >= args.min_hits:
            events_with_hits.append(i)
        if len(events_with_hits) >= need:
            break
    if len(events_with_hits) < need:
        sys.exit(f"[ERROR] only {len(events_with_hits)} events pass --min-hits={args.min_hits}, need {need}.")

    fit_idx = events_with_hits[:args.n_fit_events]
    train_idx = events_with_hits[args.n_fit_events:args.n_fit_events + args.n_train_events]
    test_idx = events_with_hits[args.n_fit_events + args.n_train_events: need]
    print(f"[INFO] fit={len(fit_idx)} train={len(train_idx)} test={len(test_idx)} events")

    stats = SPHENIX_STATS if args.norm == "sphenix" else compute_star_stats(features_mmap, args.norm_sample_events)

    print("[INFO] Fitting STAR-adapted Voxelizer bins...")
    fit_eta, fit_phi, fit_r_raw, fit_r_n = [], [], [], []
    for i in fit_idx:
        feat = np.asarray(features_mmap[i])
        _, eta_n, phi_n, r_n, r_raw = event_polar(feat, stats)
        fit_eta.append(eta_n); fit_phi.append(phi_n); fit_r_raw.append(r_raw); fit_r_n.append(r_n)
    fit_eta, fit_phi = np.concatenate(fit_eta), np.concatenate(fit_phi)
    fit_r_raw, fit_r_n = np.concatenate(fit_r_raw), np.concatenate(fit_r_n)
    final_bins = compute_star_final_bins(fit_eta, fit_phi, fit_r_raw, fit_r_n,
                                          n_bins=tuple(args.n_bins), n_radial_groups=args.n_radial_groups)
    voxelizer = build_voxelizer(final_bins, n_bins=tuple(args.n_bins),
                                 dim_sweep_order=tuple(args.dim_sweep_order),
                                 revert_order=tuple(args.revert_order))
    print(f"[INFO] Voxelizer fitted (dim_sweep_order={args.dim_sweep_order}, revert_order={args.revert_order}).")

    cache_root = os.path.expanduser(args.feature_cache_dir) if args.feature_cache_dir else tempfile.mkdtemp(prefix="fm4npp_trackfit_cache_")
    os.makedirs(cache_root, exist_ok=True)
    print(f"[INFO] Disk-caching frozen features under {cache_root}.")

    def collect_features(idx_list, model, cache_subdir):
        cache_dir = os.path.join(cache_root, cache_subdir)
        os.makedirs(cache_dir, exist_ok=True)
        paths, reg_list, pid_list = [], [], []
        for i in idx_list:
            feat = np.asarray(features_mmap[i])
            reg = np.asarray(reg_mmap[i])
            pid = np.asarray(pid_mmap[i])
            E_n, eta_n, phi_n, r_n, r_raw = event_polar(feat, stats)
            order = hierarchical_raster_scan_order(voxelizer, eta_n, phi_n, r_n)
            stack = get_frozen_feature_stack(model, E_n, eta_n, phi_n, r_n, order)
            path = os.path.join(cache_dir, f"{i}.pt")
            save_feature_stack(stack, path)
            paths.append(path); reg_list.append(reg); pid_list.append(pid)
        return paths, reg_list, pid_list

    results = {}
    for arm_label, load_ckpt in [("pretrained", True), ("random_init", False)]:
        print(f"\n[INFO] === {arm_label} ===")
        model = build_model(args.embed_dim, args.num_layers, args.d_state, args.d_conv, args.expand,
                            args.klen, args.checkpoint, load_ckpt)

        print(f"[INFO] Collecting frozen features ({args.num_layers} layers) for {len(train_idx)} train events...")
        train_paths, train_reg, train_pid = collect_features(train_idx, model, f"{arm_label}_train")

        all_train_reg = np.concatenate([r[np.asarray(p) != UNKNOWN_CLASS][:, :3]
                                         for r, p in zip(train_reg, train_pid)])
        target_mean = all_train_reg.mean(axis=0)
        target_std = all_train_reg.std(axis=0) + 1e-6
        print(f"[INFO] Target normalization (train set, valid hits only): "
              f"mean={dict(zip(TARGET_NAMES, target_mean.round(3)))}, "
              f"std={dict(zip(TARGET_NAMES, target_std.round(3)))}")

        down_model = build_head(args, device)
        n_params = sum(p.numel() for p in down_model.parameters())
        print(f"[INFO] {args.head_type} track-fit head built: {n_params:,} trainable params, "
              f"{len(TARGET_NAMES)} regression outputs ({TARGET_NAMES}).")

        print(f"[INFO] Training track-fit head for {args.head_epochs} epochs (backbone frozen throughout)...")
        down_model = train_head(down_model, train_paths, train_reg, train_pid,
                                 target_mean, target_std, args.head_epochs, args.head_lr, device)
        shutil.rmtree(os.path.join(cache_root, f"{arm_label}_train"), ignore_errors=True)

        print(f"[INFO] Collecting frozen features for {len(test_idx)} test events...")
        test_paths, test_reg, test_pid = collect_features(test_idx, model, f"{arm_label}_test")

        all_rms, all_bias = [], []
        test_records = []
        for event_idx, feat_path, reg, pid in zip(test_idx, test_paths, test_reg, test_pid):
            feat_stack = load_feature_stack(feat_path)
            r = score_event(down_model, feat_stack, reg, pid, target_mean, target_std, device)
            if r is None:
                continue
            all_rms.append(r["rms"]); all_bias.append(r["bias"])
            if args.results_dir:
                raw_feat = np.asarray(features_mmap[event_idx])
                test_records.append(dict(event_idx=int(event_idx), feat=raw_feat, reg=np.asarray(reg)[:, :3],
                                          pid=np.asarray(pid), pred_phys=r["pred_phys"], valid=r["valid"],
                                          rms=r["rms"], bias=r["bias"], n_valid=r["n_valid"], n_hits=r["n_hits"]))
        shutil.rmtree(os.path.join(cache_root, f"{arm_label}_test"), ignore_errors=True)

        all_rms = np.array(all_rms); all_bias = np.array(all_bias)  # (n_events, 3)
        results[arm_label] = dict(rms=all_rms, bias=all_bias, target_mean=target_mean, target_std=target_std)
        print(f"[RESULT] {arm_label}: " + ", ".join(
            f"{name} RMS={all_rms[:, j].mean():.4f}" for j, name in enumerate(TARGET_NAMES)
        ) + f"  (n={len(all_rms)} events)")

        if args.results_dir:
            save_arm_results(os.path.expanduser(args.results_dir), arm_label, down_model, model,
                              is_random_init=(arm_label == "random_init"), test_records=test_records)

    print("\n=== SUMMARY (per-event mean RMS residual, lower is better) ===")
    summary = {}
    for j, name in enumerate(TARGET_NAMES):
        p_arr, r_arr = results["pretrained"]["rms"][:, j], results["random_init"]["rms"][:, j]
        n = min(len(p_arr), len(r_arr))
        p_arr, r_arr = p_arr[:n], r_arr[:n]
        deltas = r_arr - p_arr  # positive delta = pretrained has LOWER (better) RMS
        print(f"\n-- {name} RMS --")
        print(f"pretrained : {p_arr.mean():.4f} (std={p_arr.std():.4f})")
        print(f"random_init: {r_arr.mean():.4f} (std={r_arr.std():.4f})")
        print(f"delta (random_init - pretrained, positive = pretrained better): {deltas.mean():+.4f} (n={n})")
        metric_summary = dict(pretrained_mean=float(p_arr.mean()), random_init_mean=float(r_arr.mean()),
                               delta_mean=float(deltas.mean()), n=int(n))
        if n >= 6:
            w_stat, w_p = sstats.wilcoxon(deltas)
            print(f"Wilcoxon signed-rank p={w_p:.4f}")
            metric_summary["wilcoxon_p"] = float(w_p)
        summary[name] = metric_summary

    if args.results_dir:
        with open(os.path.join(os.path.expanduser(args.results_dir), "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for j, (ax, name) in enumerate(zip(axes, TARGET_NAMES)):
        p_arr, r_arr = results["pretrained"]["rms"][:, j], results["random_init"]["rms"][:, j]
        ax.boxplot([p_arr, r_arr], tick_labels=["pretrained", "random-init"])
        ax.set_title(f"{name} RMS residual\n(n={min(len(p_arr), len(r_arr))} test events)")
        ax.set_ylabel(f"RMS({name}) [physical units]")
    fig.suptitle(f"Track-fitting head ({args.head_type}), per-hit kinematics regression")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\n[DONE] Saved summary figure to {args.out}")
    if not args.feature_cache_dir:
        shutil.rmtree(cache_root, ignore_errors=True)


if __name__ == "__main__":
    main()
