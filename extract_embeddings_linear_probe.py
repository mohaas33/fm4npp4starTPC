#!/usr/bin/env python3
"""
Updated probe, addressing two gaps found by reading the FM4NPP paper's
appendix directly against the actual repo code:

1. Real point ordering. Uses star_voxelizer.py's STAR-adapted Hierarchical
   Raster Scan (computed from STAR data itself, since the original
   sPHENIX-specific bin-stat .pkl files were never published) instead of the
   plain r-sort approximation the earlier probe used.

2. Trained linear probe, not raw embeddings. The paper (Q3 / Figure 8)
   states explicitly that RAW frozen FM embeddings show "no clear separation
   among particle tracks" even on the model's own native, in-distribution
   sPHENIX data -- separation only emerges "after applying a single linear
   projection." The earlier probe measured raw embeddings directly, which per
   the paper's own findings was never expected to show clustering regardless
   of domain transfer. This script trains that single linear layer (per the
   paper's own adaptation methodology) on STAR TRAIN events, then evaluates
   silhouette separation on held-out STAR TEST events -- for both the
   pretrained backbone and a random-init control, so the comparison is fair.

Simplification vs. the paper: the paper's real downstream head is a
transformer-decoder instance-segmentation adapter trained with Hungarian
matching + Dice/Focal/classification losses (Figure 4). This script instead
trains the single linear layer directly with a supervised metric-learning
objective (pull same-track embeddings together, push different-track
embeddings apart) -- a lighter-weight stand-in that targets exactly what
silhouette score measures, not a reimplementation of their full adapter.
Treat this as a probe, not a claim of matching their reported numbers.

Usage:
    python extract_embeddings_linear_probe.py \
        --data-dir data/star_fm4npp_2k \
        --checkpoint checkpoints/pp_nerf_m3_k30.ckpt \
        --n-fit-events 60 --n-train-events 300 --n-test-events 100 \
        --norm star --probe-epochs 30 \
        --out linear_probe_summary.png
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# reuse normalization/model-loading logic already validated in this repo
from extract_embeddings_pca import (
    cartesian_to_polar, polar_and_normalize, compute_star_stats, SPHENIX_STATS,
)
from star_voxelizer import compute_star_final_bins, build_voxelizer, hierarchical_raster_scan_order


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="train", choices=["train", "test"],
                    help="Which RaggedMmap split to draw ALL events from (fit/train/test subsets below "
                         "are then carved out of this split by index -- keeps this separate from the "
                         "conversion script's own train/test split, since we need our own fit/train/test "
                         "partition for the voxelizer-fitting / linear-probe-training / evaluation stages).")
    p.add_argument("--n-fit-events", type=int, default=60,
                    help="Events used ONLY to fit the STAR Voxelizer bins (eta/phi/radius binning).")
    p.add_argument("--n-train-events", type=int, default=300,
                    help="Events used to train the linear probe (separate from fit and test events).")
    p.add_argument("--n-test-events", type=int, default=100,
                    help="Held-out events for final silhouette evaluation.")
    p.add_argument("--min-hits", type=int, default=15)
    p.add_argument("--norm", default="star", choices=["star", "sphenix"])
    p.add_argument("--norm-sample-events", type=int, default=200)
    p.add_argument("--n-bins", type=int, nargs=3, default=[8, 8, 6], metavar=("ETA", "PHI", "RADIUS"))
    p.add_argument("--n-radial-groups", type=int, default=2,
                    help="STAR's TPC has a 2-group inner/outer sector structure, unlike sPHENIX's 3.")
    p.add_argument("--dim-sweep-order", type=int, nargs=3, default=[0, 1, 2], metavar=("D0", "D1", "D2"),
                    help="Voxelizer dim_sweep_order. Default [0,1,2] is this repo's original guess (unconfirmed "
                         "class default). FM4NPP's own public configs (dataset_pretrain.py's orderdict, both "
                         "mamba_pretrain.yaml and mamba_tracking.yaml default to order='EPR') use [2,1,0] -- "
                         "pass '--dim-sweep-order 2 1 0' to match that instead.")
    p.add_argument("--revert-order", type=int, nargs=3, default=[0, 1, 2], metavar=("D0", "D1", "D2"),
                    help="Voxelizer revert_order. FM4NPP's published order='EPR' preset pairs with [2,1,0] here too.")
    p.add_argument("--probe-dim", type=int, default=64, help="Output dimension of the trained linear probe.")
    p.add_argument("--probe-epochs", type=int, default=30)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=1.0, help="Triplet-loss margin for the probe training.")
    p.add_argument("--out", default="linear_probe_summary.png")

    p.add_argument("--seed", type=int, default=42,
                   help="Random seed, controls random-init backbone weights and linear probe init/training.")
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--d-state", type=int, default=32)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--klen", type=int, default=30)
    return p.parse_args()


def event_polar(feat_exyz, stats):
    """Return normalized (eta_n, phi_n, r_n), raw r, for one event's (E,x,y,z) array."""
    x, y, z = feat_exyz[:, 1], feat_exyz[:, 2], feat_exyz[:, 3]
    eta, phi, r_raw = cartesian_to_polar(x, y, z)
    # NOTE: normalized eta/phi/r computed directly here (NOT via polar_and_normalize(), which also
    # r-sorts internally) -- we need values aligned to the ORIGINAL hit order for voxelizer
    # tokenizing/fitting; reordering happens later via hierarchical_raster_scan_order().
    E_n = (feat_exyz[:, 0] - stats["E_mean"]) / stats["E_std"]
    eta_n = (eta - stats["eta_lim"][0]) / (stats["eta_lim"][1] - stats["eta_lim"][0])
    phi_n = (phi - stats["phi_lim"][0]) / (stats["phi_lim"][1] - stats["phi_lim"][0])
    r_n = (r_raw - stats["r_lim"][0]) / (stats["r_lim"][1] - stats["r_lim"][0])
    return E_n.astype(np.float32), eta_n.astype(np.float32), phi_n.astype(np.float32), \
        r_n.astype(np.float32), r_raw.astype(np.float32)


def build_model(embed_dim, num_layers, d_state, d_conv, expand, klen, checkpoint_path, load_checkpoint):
    from fm4npp.models.mambagpt import MambaGPT
    m = MambaGPT(embed_dim=embed_dim, num_layers=num_layers, d_state=d_state, d_conv=d_conv,
                 expand=expand, klen=klen, dropout=0.0, embed_method='add', pe_method='nerf').cuda()
    if load_checkpoint:
        ckpt = torch.load(os.path.expanduser(checkpoint_path), map_location='cpu', weights_only=False)
        state_dict = {k.replace('module.', '', 1): v for k, v in ckpt['model_state'].items()}
        missing, unexpected = m.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[WARN] missing={missing} unexpected={unexpected}")
    m.eval()
    for p_ in m.parameters():
        p_.requires_grad = False
    return m


def get_frozen_embedding(model, E_n, eta_n, phi_n, r_n, order):
    """Reorder one event's points per the Hierarchical Raster Scan order, run the frozen backbone,
    return (N, embed_dim) embeddings back in ORIGINAL point order (so labels still line up)."""
    feat_ordered = np.stack([E_n[order], eta_n[order], phi_n[order], r_n[order]], axis=-1).astype(np.float32)
    x_in = torch.from_numpy(feat_ordered).unsqueeze(0).cuda()
    with torch.no_grad():
        _, feature_layers, _ = model(x_in, return_z=True)
    emb_ordered = feature_layers[-1].squeeze(0).cpu().numpy()  # (N, embed_dim), in ORDERED sequence
    emb_original_order = np.empty_like(emb_ordered)
    emb_original_order[order] = emb_ordered  # undo the reordering so index i matches original track_id[i]
    return emb_original_order


class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)


def triplet_loss_batch(emb, track_ids, margin):
    """Vectorized online triplet loss: hardest positive + hardest negative per anchor, no Python loop."""
    emb = nn.functional.normalize(emb, dim=-1)
    dist = torch.cdist(emb, emb, p=2)
    track_ids = torch.as_tensor(np.asarray(track_ids).copy(), device=emb.device)

    same = track_ids.unsqueeze(0) == track_ids.unsqueeze(1)
    diff = track_ids.unsqueeze(0) != track_ids.unsqueeze(1)
    eye = torch.eye(len(track_ids), dtype=torch.bool, device=emb.device)
    same = same & ~eye

    valid_anchor = same.any(dim=1) & diff.any(dim=1)
    if valid_anchor.sum() == 0:
        return None

    hardest_pos = dist.masked_fill(~same, float('-inf')).max(dim=1).values
    hardest_neg = dist.masked_fill(~diff, float('inf')).min(dim=1).values

    losses = torch.clamp(hardest_pos - hardest_neg + margin, min=0.0)[valid_anchor]
    return losses.mean()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    from mmap_ninja import RaggedMmap
    from sklearn.metrics import silhouette_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sstats

    feat_path = os.path.join(os.path.expanduser(args.data_dir), f"features_{args.split}")
    tgt_path = os.path.join(os.path.expanduser(args.data_dir), f"seg_target_{args.split}")
    features_mmap = RaggedMmap(feat_path)
    target_mmap = RaggedMmap(tgt_path)
    n_avail = len(features_mmap)
    print(f"[INFO] {n_avail} events available in split '{args.split}'")

    need = args.n_fit_events + args.n_train_events + args.n_test_events
    if need > n_avail:
        sys.exit(f"[ERROR] need {need} events (fit+train+test) but only {n_avail} available. "
                 f"Reduce --n-fit-events/--n-train-events/--n-test-events or convert more events.")

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

    # --- normalization stats ---
    stats = SPHENIX_STATS if args.norm == "sphenix" else compute_star_stats(features_mmap, args.norm_sample_events)

    # --- fit the STAR-adapted voxelizer on the fit-only events ---
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

    def collect(idx_list, model):
        embs, labels = [], []
        for i in idx_list:
            feat = np.asarray(features_mmap[i])
            track_id = np.asarray(target_mmap[i])
            E_n, eta_n, phi_n, r_n, r_raw = event_polar(feat, stats)
            order = hierarchical_raster_scan_order(voxelizer, eta_n, phi_n, r_n)
            emb = get_frozen_embedding(model, E_n, eta_n, phi_n, r_n, order)
            embs.append(emb); labels.append(track_id)
        return embs, labels  # lists, per-event (variable N)

    def train_probe(embs, labels, in_dim, out_dim, epochs, lr, margin):
        probe = LinearProbe(in_dim, out_dim).cuda()
        opt = torch.optim.Adam(probe.parameters(), lr=lr)
        for ep in range(epochs):
            total_loss, n_batches = 0.0, 0
            for emb, lab in zip(embs, labels):
                if len(np.unique(lab)) < 2:
                    continue
                x = torch.from_numpy(emb).float().cuda()
                out = probe(x)
                loss = triplet_loss_batch(out, lab, margin)
                if loss is None:
                    continue
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item(); n_batches += 1
            if n_batches and (ep % 5 == 0 or ep == epochs - 1):
                print(f"    [probe] epoch {ep}: mean triplet loss = {total_loss / n_batches:.4f}")
        probe.eval()
        return probe

    def eval_silhouette(probe, embs, labels):
        scores = []
        with torch.no_grad():
            for emb, lab in zip(embs, labels):
                n_unique = len(np.unique(lab))
                if n_unique < 2 or emb.shape[0] <= n_unique:
                    continue
                x = torch.from_numpy(emb).float().cuda()
                out = probe(x).cpu().numpy()
                scores.append(silhouette_score(out, lab))
        return np.array(scores)

    results = {}
    for label, load_ckpt in [("pretrained", True), ("random_init", False)]:
        print(f"\n[INFO] === {label} ===")
        model = build_model(args.embed_dim, args.num_layers, args.d_state, args.d_conv, args.expand,
                            args.klen, args.checkpoint, load_ckpt)

        print(f"[INFO] Collecting frozen embeddings for {len(train_idx)} train events...")
        train_embs, train_labels = collect(train_idx, model)
        print(f"[INFO] Training linear probe ({args.probe_dim}-d) for {args.probe_epochs} epochs...")
        probe = train_probe(train_embs, train_labels, args.embed_dim, args.probe_dim,
                            args.probe_epochs, args.probe_lr, args.margin)

        print(f"[INFO] Collecting frozen embeddings for {len(test_idx)} test events...")
        test_embs, test_labels = collect(test_idx, model)
        sil_scores = eval_silhouette(probe, test_embs, test_labels)
        results[label] = sil_scores
        print(f"[RESULT] {label}: mean silhouette (post-linear-probe, held-out test) = "
              f"{sil_scores.mean():.4f} +/- {sil_scores.std():.4f}  (n={len(sil_scores)} valid events)")

    p_scores, r_scores = results["pretrained"], results["random_init"]
    n = min(len(p_scores), len(r_scores))
    p_scores, r_scores = p_scores[:n], r_scores[:n]
    deltas = p_scores - r_scores

    print("\n=== SUMMARY (post-linear-probe silhouette, held-out test events) ===")
    print(f"Mean, pretrained : {p_scores.mean():.4f} (std={p_scores.std():.4f})")
    print(f"Mean, random-init: {r_scores.mean():.4f} (std={r_scores.std():.4f})")
    print(f"Mean delta: {deltas.mean():+.4f} (std={deltas.std():.4f}, n={n})")
    if n >= 6:
        w_stat, w_p = sstats.wilcoxon(deltas)
        print(f"Wilcoxon signed-rank test on deltas: p={w_p:.4f}")
        print(">>> p < 0.05: gap unlikely to be noise." if w_p < 0.05
              else ">>> p >= 0.05: cannot rule out noise at this sample size.")
    else:
        print("[INFO] fewer than 6 paired events -- skipping significance test.")

    fig, (axb, axh) = plt.subplots(1, 2, figsize=(11, 4.5))
    axb.boxplot([p_scores, r_scores], tick_labels=["pretrained\n+linear probe", "random-init\n+linear probe"])
    axb.set_ylabel("silhouette score (post-linear-probe)")
    axb.set_title(f"Post-linear-probe silhouette\n(n={n} held-out test events, norm={args.norm})")
    axh.hist(deltas, bins=min(20, max(5, n // 3)), color="steelblue", edgecolor="black")
    axh.axvline(0, color="red", linestyle="--", label="zero (no effect)")
    axh.axvline(deltas.mean(), color="black", label=f"mean = {deltas.mean():+.4f}")
    axh.set_xlabel("delta (pretrained - random-init), per event")
    axh.set_title("Per-event delta, post-linear-probe")
    axh.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\n[DONE] Saved summary figure to {args.out}")


if __name__ == "__main__":
    main()