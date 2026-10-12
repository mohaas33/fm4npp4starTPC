#!/usr/bin/env python3
"""
STAR-adapted PID (particle species classification) evaluation, mirroring
eval_track_finding_star.py's pattern but for FM4NPP's OTHER real downstream
head: train/downstream/model.py's AttentionHead/MambaHead (used by
point_classification_trainer.py), a much simpler per-point classifier than
tracking's MambaAttentionHead -- no Hungarian matching, no query prototypes,
just per-hit class logits from the same frozen-backbone weighted-layer-average
input convention (pretrain=True).

Requires real per-hit species labels -- only available via
convert_star_pythia_multifile.py's pid_target (built from the verified
T<->TT truth join on /home/daq/Work/STAR/files/pythia8_pp510_minbias/).
The earlier single-big-file Pythia dataset has NO usable PID truth (gePid
uniformly -1 there) -- this script will silently produce garbage if pointed
at that data's pid_target (all UNKNOWN_CLASS).

Classes (must match convert_star_pythia_multifile.py's PID_CLASS_NAMES):
    0 pion, 1 muon, 2 electron, 3 proton, 4 kaon, 5 other, 6 unknown
Class 6 "unknown" (no T<->TT truth match for that hit's track) is EXCLUDED
from both training loss and evaluation metrics -- there is no real label to
learn or score against for those hits.

Usage:
    python eval_pid_star.py \
        --data-dir data/pythia510_multifile_8k \
        --checkpoint checkpoints/pp_nerf_m3_k30.ckpt \
        --n-fit-events 60 --n-train-events 3700 --n-test-events 200 \
        --head-epochs 30 --seed 42 \
        --results-dir results/pid_pythia_multifile \
        --events-out pid_pythia_multifile_events.png \
        --out pid_pythia_multifile_summary.png
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
# reuse the already-built, already-verified backbone/disk-cache plumbing instead of duplicating it
from eval_track_finding_star import get_frozen_feature_stack, save_feature_stack, load_feature_stack

FM4NPP_REPO = os.path.expanduser(os.environ.get("FM4NPP_REPO", "~/Work/STAR/PP_collision"))
sys.path.append(FM4NPP_REPO)
sys.path.append(os.path.join(FM4NPP_REPO, "train", "downstream"))
from model import AttentionHead, MambaHead

PID_CLASS_NAMES = ["pion", "muon", "electron", "proton", "kaon", "other", "unknown"]
UNKNOWN_CLASS = 6  # must match convert_star_pythia_multifile.py -- excluded from loss & metrics
PID_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red",
              4: "tab:purple", 5: "tab:brown", 6: "lightgray"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--n-fit-events", type=int, default=60)
    p.add_argument("--n-train-events", type=int, default=300)
    p.add_argument("--n-test-events", type=int, default=100)
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

    p.add_argument("--head-type", default="attention", choices=["attention", "mamba"],
                    help="FM4NPP's two real point-classification head variants (model.py).")
    p.add_argument("--head-embed-dim", type=int, default=256)
    p.add_argument("--head-layers", type=int, default=1, help="Matches FM4NPP's own point_classification_trainer.py default.")
    p.add_argument("--head-dropout", type=float, default=0.1)
    p.add_argument("--head-epochs", type=int, default=15)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--class-weighted", action="store_true", default=True,
                    help="Weight the cross-entropy loss by inverse train-set class frequency (pion-dominated "
                         "sample otherwise biases heavily toward predicting 'pion').")
    p.add_argument("--no-class-weighted", dest="class_weighted", action="store_false")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="pid_head_summary.png")
    p.add_argument("--feature-cache-dir", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--events-out", default=None)
    p.add_argument("--n-display-events", type=int, default=6)
    return p.parse_args()


def build_head(args, device):
    Head = AttentionHead if args.head_type == "attention" else MambaHead
    kwargs = dict(input_dim=args.embed_dim, embed_dim=args.head_embed_dim, num_layers=args.head_layers,
                  num_feature_layers=args.num_layers, num_output_dim=len(PID_CLASS_NAMES) - 1,  # exclude "unknown"
                  dropout=args.head_dropout)
    if args.head_type == "attention":
        kwargs["num_heads"] = 4
    else:
        kwargs.update(d_state=64, d_conv=4, expand=2)
    return Head(**kwargs).to(device)


def forward_event(down_model, feature_stack_cpu, device):
    feature = feature_stack_cpu.to(device).unsqueeze(1)  # (num_layers, 1, N, embed_dim)
    n_hits = feature.shape[2]
    dummy_x = torch.zeros(1, n_hits, 4, device=device)
    padding_mask = torch.ones(1, n_hits, dtype=torch.bool, device=device)
    pred_dict = down_model(dummy_x, feature, padding_mask=padding_mask, pretrain=True)
    return pred_dict["pred_logits"].squeeze(0)  # (N, num_classes_known)


def masked_ce_loss(logits, labels, class_weights, device):
    """Cross-entropy over per-hit class logits, excluding UNKNOWN_CLASS hits (no real label)."""
    valid = labels != UNKNOWN_CLASS
    if valid.sum() == 0:
        return None
    w = torch.as_tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
    return F.cross_entropy(logits[valid], labels[valid].to(device), weight=w)


def train_head(down_model, feature_paths, labels_list, epochs, lr, device, class_weights):
    opt = torch.optim.AdamW(down_model.parameters(), lr=lr, weight_decay=1e-4)
    down_model.train()
    for ep in range(epochs):
        total, n_seen = 0.0, 0
        order = np.random.permutation(len(feature_paths))
        for idx in order:
            feat_stack = load_feature_stack(feature_paths[idx])
            labels = torch.as_tensor(labels_list[idx], dtype=torch.long)
            if (labels != UNKNOWN_CLASS).sum() == 0:
                continue
            logits = forward_event(down_model, feat_stack, device)
            loss = masked_ce_loss(logits, labels, class_weights, device)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n_seen += 1
        if n_seen and (ep % 2 == 0 or ep == epochs - 1):
            print(f"    [pid-head] epoch {ep}: mean masked CE loss = {total / n_seen:.4f}")
    down_model.eval()
    return down_model


def score_event(down_model, feat_stack, labels, device):
    from sklearn.metrics import f1_score, balanced_accuracy_score
    with torch.no_grad():
        logits = forward_event(down_model, feat_stack, device)
        pred = logits.argmax(dim=-1).cpu().numpy()
    labels = np.asarray(labels)
    valid = labels != UNKNOWN_CLASS
    if valid.sum() == 0:
        return None
    y_true, y_pred = labels[valid], pred[valid]
    present = np.unique(y_true)
    macro_f1 = f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = float((y_true == y_pred).mean())
    return dict(macro_f1=macro_f1, balanced_accuracy=bal_acc, accuracy=acc,
                pred=pred, valid=valid, n_valid=int(valid.sum()), n_hits=len(labels))


def save_arm_results(results_dir, arm_label, down_model, backbone_model, is_random_init, test_records):
    arm_dir = os.path.join(results_dir, arm_label)
    os.makedirs(arm_dir, exist_ok=True)
    torch.save(down_model.state_dict(), os.path.join(arm_dir, "down_model.pt"))
    if is_random_init:
        torch.save(backbone_model.state_dict(), os.path.join(arm_dir, "backbone.pt"))
    with open(os.path.join(arm_dir, "test_results.pkl"), "wb") as f:
        pickle.dump(test_records, f)
    print(f"[INFO] Saved {arm_label} PID results to {arm_dir}/")


def select_display_events(test_records, n_events, forced_event_ids=None):
    """If forced_event_ids is given, use exactly those events (in that order) instead of
    re-ranking by this call's own macro-F1 -- needed so pretrained and random_init show the
    SAME events; picking independently per arm means their "worst/median/best" picks land on
    different events, since each arm's macro-F1 ranking differs (same issue found and fixed in
    eval_track_finding_star.py's event displays)."""
    if forced_event_ids is not None:
        by_id = {r["event_idx"]: r for r in test_records}
        return [by_id[eid] for eid in forced_event_ids if eid in by_id]
    order = np.argsort([r["macro_f1"] for r in test_records])
    n = len(order)
    if n <= n_events:
        return [test_records[i] for i in order]
    pick_pos = sorted(set(np.linspace(0, n - 1, n_events).round().astype(int).tolist()))
    return [test_records[order[p]] for p in pick_pos]


def plot_event_displays(test_records, out_path, n_events, arm_label, forced_event_ids=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # last row is a color-legend panel, not an event -- so with n_events=8 total rows, 7 are real events
    n_event_rows = max(1, n_events - 1)
    picks = select_display_events(test_records, n_event_rows, forced_event_ids=forced_event_ids)
    n_rows = len(picks) + 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.6 * n_rows), squeeze=False)
    legend_handles = [mpatches.Patch(color=PID_COLORS[c], label=PID_CLASS_NAMES[c]) for c in range(7)]
    for row, rec in enumerate(picks):
        feat, true_labels, pred, valid = rec["feat"], rec["true_labels"], rec["pred_full"], rec["valid_mask"]
        cx, cy = feat[:, 1], feat[:, 2]

        ax = axes[row, 0]
        for c in range(7):
            m = true_labels == c
            if m.any():
                ax.scatter(cx[m], cy[m], s=12, color=PID_COLORS[c])
        ax.set_title(f"event {rec['event_idx']}: TRUE species")

        ax = axes[row, 1]
        # predicted only defined for valid (non-unknown) hits; show unknown hits in their true-gray too
        for c in range(6):
            m = valid & (pred == c)
            if m.any():
                ax.scatter(cx[m], cy[m], s=12, color=PID_COLORS[c])
        ax.scatter(cx[~valid], cy[~valid], s=12, color=PID_COLORS[6])
        ax.set_title(f"PREDICTED (macro-F1={rec['macro_f1']:.3f})")

        ax = axes[row, 2]
        correct = valid & (pred == true_labels)
        wrong = valid & (pred != true_labels)
        ax.scatter(cx, cy, s=10, color="lightsteelblue", zorder=1)
        ax.scatter(cx[wrong], cy[wrong], s=28, color="red", marker="x", label="wrong", zorder=2)
        ax.scatter(cx[correct], cy[correct], s=28, color="green", marker="x", label="correct", zorder=3)
        ax.set_title(f"OUTCOME (acc={rec['accuracy']:.3f}, n_valid={rec['n_valid']}/{rec['n_hits']})")
        if row == 0:
            ax.legend(fontsize=7, loc="upper right")

        for a in axes[row]:
            a.set_xlabel("x [cm]"); a.set_ylabel("y [cm]"); a.set_aspect("equal", adjustable="datalim")

    # legend panel, replacing what would have been one more event row
    legend_row = n_rows - 1
    for a in axes[legend_row]:
        a.axis("off")
    axes[legend_row, 1].legend(handles=legend_handles, loc="center", ncol=2, fontsize=11,
                                frameon=True, title="species color legend")

    fig.suptitle(f"{arm_label}: PID event displays, spread across macro-F1 range (not cherry-picked)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[INFO] Saved PID event-display figure to {out_path}")
    return [rec["event_idx"] for rec in picks]  # so the caller can reuse the same picks for other arms


def plot_confusion_matrix(all_true, all_pred, out_path, arm_label):
    from sklearn.metrics import confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(range(6))  # exclude "unknown" -- never a true OR predicted class here
    cm = confusion_matrix(all_true, all_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(6)); ax.set_xticklabels(PID_CLASS_NAMES[:6], rotation=45, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(PID_CLASS_NAMES[:6])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                     color="white" if cm[i, j] < 0.5 else "black", fontsize=8)
    fig.colorbar(im, label="row-normalized fraction")
    ax.set_title(f"{arm_label}: PID confusion matrix (pooled test hits)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[INFO] Saved confusion matrix to {out_path}")


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
    tgt_path = os.path.join(os.path.expanduser(args.data_dir), f"seg_target_{args.split}")
    pid_path = os.path.join(os.path.expanduser(args.data_dir), f"pid_target_{args.split}")
    features_mmap = RaggedMmap(feat_path)
    seg_mmap = RaggedMmap(tgt_path)
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

    cache_root = os.path.expanduser(args.feature_cache_dir) if args.feature_cache_dir else tempfile.mkdtemp(prefix="fm4npp_pid_cache_")
    os.makedirs(cache_root, exist_ok=True)
    print(f"[INFO] Disk-caching frozen features under {cache_root}.")

    def collect_features(idx_list, model, cache_subdir):
        cache_dir = os.path.join(cache_root, cache_subdir)
        os.makedirs(cache_dir, exist_ok=True)
        paths, pid_labels = [], []
        for i in idx_list:
            feat = np.asarray(features_mmap[i])
            pid_label = np.asarray(pid_mmap[i])
            E_n, eta_n, phi_n, r_n, r_raw = event_polar(feat, stats)
            order = hierarchical_raster_scan_order(voxelizer, eta_n, phi_n, r_n)
            stack = get_frozen_feature_stack(model, E_n, eta_n, phi_n, r_n, order)
            path = os.path.join(cache_dir, f"{i}.pt")
            save_feature_stack(stack, path)
            paths.append(path)
            pid_labels.append(pid_label)
        return paths, pid_labels

    results = {}
    shared_display_event_ids = None  # set after the first arm, reused so both arms show the SAME events
    for arm_label, load_ckpt in [("pretrained", True), ("random_init", False)]:
        print(f"\n[INFO] === {arm_label} ===")
        model = build_model(args.embed_dim, args.num_layers, args.d_state, args.d_conv, args.expand,
                            args.klen, args.checkpoint, load_ckpt)

        print(f"[INFO] Collecting frozen features ({args.num_layers} layers) for {len(train_idx)} train events...")
        train_paths, train_pid = collect_features(train_idx, model, f"{arm_label}_train")

        class_weights = None
        if args.class_weighted:
            all_train_pid = np.concatenate(train_pid)
            counts = np.array([max((all_train_pid == c).sum(), 1) for c in range(6)])
            class_weights = (counts.sum() / (6 * counts)).astype(np.float32)
            print(f"[INFO] Class weights (inverse frequency, train set): "
                  f"{dict(zip(PID_CLASS_NAMES[:6], class_weights.round(2)))}")

        down_model = build_head(args, device)
        n_params = sum(p.numel() for p in down_model.parameters())
        print(f"[INFO] {args.head_type} PID head built: {n_params:,} trainable params, "
              f"{len(PID_CLASS_NAMES) - 1} output classes.")

        print(f"[INFO] Training PID head for {args.head_epochs} epochs (backbone frozen throughout)...")
        down_model = train_head(down_model, train_paths, train_pid, args.head_epochs, args.head_lr, device, class_weights)
        shutil.rmtree(os.path.join(cache_root, f"{arm_label}_train"), ignore_errors=True)

        print(f"[INFO] Collecting frozen features for {len(test_idx)} test events...")
        test_paths, test_pid = collect_features(test_idx, model, f"{arm_label}_test")

        macro_f1s, bal_accs, accs = [], [], []
        test_records, pooled_true, pooled_pred = [], [], []
        for event_idx, feat_path, labels in zip(test_idx, test_paths, test_pid):
            feat_stack = load_feature_stack(feat_path)
            r = score_event(down_model, feat_stack, labels, device)
            if r is None:
                continue
            macro_f1s.append(r["macro_f1"]); bal_accs.append(r["balanced_accuracy"]); accs.append(r["accuracy"])
            labels_arr = np.asarray(labels)
            pooled_true.append(labels_arr[r["valid"]])
            pooled_pred.append(r["pred"][r["valid"]])
            if args.results_dir or args.events_out:
                raw_feat = np.asarray(features_mmap[event_idx])
                test_records.append(dict(
                    event_idx=int(event_idx), feat=raw_feat, true_labels=labels_arr,
                    pred_full=r["pred"], valid_mask=r["valid"],
                    macro_f1=r["macro_f1"], balanced_accuracy=r["balanced_accuracy"], accuracy=r["accuracy"],
                    n_valid=r["n_valid"], n_hits=r["n_hits"],
                ))
        shutil.rmtree(os.path.join(cache_root, f"{arm_label}_test"), ignore_errors=True)

        results[arm_label] = dict(macro_f1=np.array(macro_f1s), balanced_accuracy=np.array(bal_accs),
                                   accuracy=np.array(accs),
                                   pooled_true=np.concatenate(pooled_true), pooled_pred=np.concatenate(pooled_pred))
        print(f"[RESULT] {arm_label}: macro-F1 = {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f}, "
              f"balanced accuracy = {np.mean(bal_accs):.4f} +/- {np.std(bal_accs):.4f}  (n={len(macro_f1s)} events)")

        if args.results_dir:
            save_arm_results(os.path.expanduser(args.results_dir), arm_label, down_model, model,
                              is_random_init=(arm_label == "random_init"), test_records=test_records)
        if args.events_out:
            base, ext = os.path.splitext(os.path.expanduser(args.events_out))
            chosen_ids = plot_event_displays(test_records, f"{base}_{arm_label}{ext}", args.n_display_events,
                                              arm_label, forced_event_ids=shared_display_event_ids)
            if shared_display_event_ids is None:
                shared_display_event_ids = chosen_ids  # lock in pretrained's picks for random_init to reuse
            plot_confusion_matrix(results[arm_label]["pooled_true"], results[arm_label]["pooled_pred"],
                                   f"{base}_{arm_label}_confusion{ext}", arm_label)

    print("\n=== SUMMARY (per-hit species classification, UNKNOWN_CLASS excluded) ===")
    summary = {}
    for metric in ("macro_f1", "balanced_accuracy", "accuracy"):
        p_arr, r_arr = results["pretrained"][metric], results["random_init"][metric]
        n = min(len(p_arr), len(r_arr))
        p_arr, r_arr = p_arr[:n], r_arr[:n]
        deltas = p_arr - r_arr
        print(f"\n-- {metric} --")
        print(f"pretrained : {p_arr.mean():.4f} (std={p_arr.std():.4f})")
        print(f"random_init: {r_arr.mean():.4f} (std={r_arr.std():.4f})")
        print(f"delta      : {deltas.mean():+.4f} (std={deltas.std():.4f}, n={n})")
        metric_summary = dict(pretrained_mean=float(p_arr.mean()), random_init_mean=float(r_arr.mean()),
                               delta_mean=float(deltas.mean()), delta_std=float(deltas.std()), n=int(n))
        if n >= 6:
            w_stat, w_p = sstats.wilcoxon(deltas)
            print(f"Wilcoxon signed-rank p={w_p:.4f}"
                  + (" -> unlikely to be noise." if w_p < 0.05 else " -> cannot rule out noise at this n."))
            metric_summary["wilcoxon_p"] = float(w_p)
        else:
            print("[INFO] fewer than 6 paired events -- skipping significance test.")
        summary[metric] = metric_summary

    if args.results_dir:
        with open(os.path.join(os.path.expanduser(args.results_dir), "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in zip(axes, ("macro_f1", "balanced_accuracy"), ("macro-F1", "balanced accuracy")):
        p_arr, r_arr = results["pretrained"][metric], results["random_init"][metric]
        ax.boxplot([p_arr, r_arr], tick_labels=["pretrained", "random-init"])
        ax.set_title(f"{title}\n(n={min(len(p_arr), len(r_arr))} test events)")
        ax.set_ylabel(title)
    fig.suptitle(f"PID head ({args.head_type}), per-hit species classification")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\n[DONE] Saved summary figure to {args.out}")
    if not args.feature_cache_dir:
        shutil.rmtree(cache_root, ignore_errors=True)


if __name__ == "__main__":
    main()
