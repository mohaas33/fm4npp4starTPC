#!/usr/bin/env python3
"""
Track-finding evaluation on top of the linear-probe embeddings.

Why this script exists: extract_embeddings_linear_probe.py reports silhouette
score, which is NOT a track-finding metric. Silhouette assumes the true
track-ID grouping is already known and only asks "are hits from the same true
track closer to each other than to hits from other true tracks, in embedding
space?" No clustering, no hit-to-track assignment, and no "reconstructed
track" ever exists anywhere in that script -- it measures separability of a
known grouping, not the ability to recover that grouping from scratch.

This script adds the missing step. For each held-out test event:

  1. Run KMeans on the post-linear-probe embeddings, with k = the TRUE number
     of tracks in that event (see caveat below -- this is a simplification).
  2. Hungarian-match predicted clusters to true tracks by shared-hit count
     (maximizes total correctly-assigned hits across the whole event).
  3. Report, per matched pair: hit efficiency (matched hits / true track size)
     and purity (matched hits / predicted cluster size). A track counts as
     "found" under the standard double-majority convention (TrackML-style):
     efficiency >= 0.5 AND purity >= 0.5.
  4. Aggregate: Adjusted Rand Index (ARI), fraction of tracks found, and
     fraction of all hits landing in a found track -- for both the pretrained
     backbone and a random-init control, paired per event, same as the
     silhouette probe.
  5. Draw actual event displays: true ("generated") hits in blue, hits that
     ended up in a found track ("reconstructed") marked with a red cross on
     top -- in real detector x/y (cm), not embedding/PCA space.

Caveats (same spirit as extract_embeddings_linear_probe.py's own docstring --
this is a probe, not a claim of a working tracking pipeline):

  - KMeans is handed the TRUE number of tracks per event. Real track-finding
    does not know this in advance -- that is a materially harder, separate
    problem. Giving KMeans the true k isolates "does the embedding separate
    tracks that are already counted" from "can you also discover how many
    tracks there are", which this script does NOT test. A density-based
    method (DBSCAN/HDBSCAN, which can also leave hits unclustered) would be
    the honest next step if this direction is pursued further -- see
    README.md's Known Limitations.
  - Still built on the triplet-loss linear probe, not the paper's own
    Hungarian-matched instance-segmentation adapter (README.md, caveat (b)).
  - ARI and "fraction of tracks found" here are NOT directly comparable to
    the FM4NPP paper's own reported ARI/efficiency numbers -- different
    adapter, different clustering step, different (STAR, not sPHENIX) data.

Usage:
    python evaluate_tracking.py \
        --data-dir data/star_fm4npp_2k \
        --checkpoint checkpoints/pp_nerf_m3_k30.ckpt \
        --n-fit-events 60 --n-train-events 300 --n-test-events 100 \
        --norm star --probe-epochs 30 --seed 1 \
        --summary-out tracking_eval_summary.png \
        --events-out tracking_eval_events.png
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from extract_embeddings_pca import (
    cartesian_to_polar, polar_and_normalize, compute_star_stats, SPHENIX_STATS,
)
from star_voxelizer import compute_star_final_bins, build_voxelizer, hierarchical_raster_scan_order
from extract_embeddings_linear_probe import (
    event_polar, build_model, get_frozen_embedding, LinearProbe, triplet_loss_batch,
)


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
    p.add_argument("--dim-sweep-order", type=int, nargs=3, default=[0, 1, 2], metavar=("D0", "D1", "D2"),
                    help="Voxelizer dim_sweep_order. Default [0,1,2] is this repo's original guess. FM4NPP's "
                         "own public configs default to order='EPR' -> [2,1,0]; pass '--dim-sweep-order 2 1 0' "
                         "to match (same correction as extract_embeddings_linear_probe.py).")
    p.add_argument("--revert-order", type=int, nargs=3, default=[0, 1, 2], metavar=("D0", "D1", "D2"),
                    help="Voxelizer revert_order. FM4NPP's order='EPR' preset pairs with [2,1,0] here too.")
    p.add_argument("--probe-dim", type=int, default=64)
    p.add_argument("--probe-epochs", type=int, default=30)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42,
                    help="Controls random-init backbone weights, probe init/training, and KMeans init.")

    p.add_argument("--double-majority-threshold", type=float, default=0.5,
                    help="Efficiency AND purity threshold for a matched cluster to count as a 'found' track "
                         "(0.5/0.5 is the standard TrackML-style double-majority convention).")
    p.add_argument("--n-plot-events", type=int, default=6,
                    help="How many example held-out test events to render in the event-display figure.")
    p.add_argument("--plot-coords", default="xy", choices=["xy", "etaphi"],
                    help="Real detector coordinates to plot hits in -- 'xy' (transverse plane, cm) or 'etaphi'.")
    p.add_argument("--summary-out", default="tracking_eval_summary.png")
    p.add_argument("--events-out", default="tracking_eval_events.png")

    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--d-state", type=int, default=32)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--klen", type=int, default=30)
    return p.parse_args()


def train_probe(embs, labels, in_dim, out_dim, epochs, lr, margin):
    """Identical to the nested function of the same name in
    extract_embeddings_linear_probe.py, hoisted to module level here since
    that script doesn't expose it for reuse."""
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


def collect(idx_list, features_mmap, target_mmap, stats, voxelizer, model):
    """Same shape of result as the nested collect() in extract_embeddings_linear_probe.py,
    but also returns the raw (E,x,y,z) feature array per event (needed for the event-display
    plot's real detector coordinates, in ORIGINAL hit order matching labels/embeddings)."""
    embs, labels, feats = [], [], []
    for i in idx_list:
        feat = np.asarray(features_mmap[i])
        track_id = np.asarray(target_mmap[i])
        E_n, eta_n, phi_n, r_n, r_raw = event_polar(feat, stats)
        order = hierarchical_raster_scan_order(voxelizer, eta_n, phi_n, r_n)
        emb = get_frozen_embedding(model, E_n, eta_n, phi_n, r_n, order)
        embs.append(emb); labels.append(track_id); feats.append(feat)
    return embs, labels, feats


def cluster_and_match(emb, true_labels, seed, threshold):
    """KMeans (k = true number of tracks) on one event's post-probe embeddings, then
    Hungarian-match predicted clusters to true tracks by shared-hit count.

    Returns None if the event can't be evaluated (too few tracks/hits), else a dict:
        pred            : (N,) predicted cluster label per hit
        ari             : Adjusted Rand Index vs. true labels
        per_track       : list of {true_id, efficiency, purity, found} per true track
        frac_found      : fraction of true tracks meeting the double-majority threshold
        correct_mask    : (N,) bool, True for hits belonging to a "found" track's matched pair
        hit_frac_found  : correct_mask.mean() -- fraction of ALL hits landing in a found track
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from scipy.optimize import linear_sum_assignment

    true_labels = np.asarray(true_labels)
    true_ids = np.unique(true_labels)
    n_tracks = len(true_ids)
    if n_tracks < 2 or emb.shape[0] <= n_tracks:
        return None

    try:
        km = KMeans(n_clusters=n_tracks, n_init=10, random_state=seed)
        pred = km.fit_predict(emb)
    except Exception as e:
        print(f"[WARN] KMeans failed on this event ({e}), skipping.")
        return None

    ari = adjusted_rand_score(true_labels, pred)

    pred_ids = np.unique(pred)
    true_idx = {t: i for i, t in enumerate(true_ids)}
    pred_idx = {c: i for i, c in enumerate(pred_ids)}
    cm = np.zeros((len(true_ids), len(pred_ids)), dtype=int)
    for t, c in zip(true_labels, pred):
        cm[true_idx[t], pred_idx[c]] += 1
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize total shared hits

    correct_mask = np.zeros(len(true_labels), dtype=bool)
    per_track = []
    n_found = 0
    for r, c in zip(row_ind, col_ind):
        true_id, pred_id = true_ids[r], pred_ids[c]
        shared = cm[r, c]
        true_size = int((true_labels == true_id).sum())
        pred_size = int((pred == pred_id).sum())
        efficiency = shared / true_size
        purity = shared / pred_size if pred_size > 0 else 0.0
        found = efficiency >= threshold and purity >= threshold
        if found:
            n_found += 1
            correct_mask |= (true_labels == true_id) & (pred == pred_id)
        per_track.append(dict(true_id=int(true_id), efficiency=float(efficiency),
                               purity=float(purity), found=bool(found)))

    return dict(pred=pred, ari=float(ari), per_track=per_track,
                frac_found=n_found / n_tracks, correct_mask=correct_mask,
                hit_frac_found=float(correct_mask.mean()))


def main():
    args = parse_args()
    from mmap_ninja import RaggedMmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sstats

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    feat_path = os.path.join(os.path.expanduser(args.data_dir), f"features_{args.split}")
    tgt_path = os.path.join(os.path.expanduser(args.data_dir), f"seg_target_{args.split}")
    features_mmap = RaggedMmap(feat_path)
    target_mmap = RaggedMmap(tgt_path)
    n_avail = len(features_mmap)
    print(f"[INFO] {n_avail} events available in split '{args.split}'")

    need = args.n_fit_events + args.n_train_events + args.n_test_events
    if need > n_avail:
        sys.exit(f"[ERROR] need {need} events (fit+train+test) but only {n_avail} available.")

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
    print("[INFO] Voxelizer fitted.")

    results = {}
    plot_material = None  # filled in on the pretrained pass, for the event-display figure
    for label, load_ckpt in [("pretrained", True), ("random_init", False)]:
        print(f"\n[INFO] === {label} ===")
        model = build_model(args.embed_dim, args.num_layers, args.d_state, args.d_conv, args.expand,
                            args.klen, args.checkpoint, load_ckpt)

        print(f"[INFO] Collecting frozen embeddings for {len(train_idx)} train events...")
        train_embs, train_labels, _ = collect(train_idx, features_mmap, target_mmap, stats, voxelizer, model)
        print(f"[INFO] Training linear probe ({args.probe_dim}-d) for {args.probe_epochs} epochs...")
        probe = train_probe(train_embs, train_labels, args.embed_dim, args.probe_dim,
                            args.probe_epochs, args.probe_lr, args.margin)

        print(f"[INFO] Collecting frozen embeddings for {len(test_idx)} test events...")
        test_embs, test_labels, test_feats = collect(test_idx, features_mmap, target_mmap, stats, voxelizer, model)

        aris, fracs_found, hit_fracs = [], [], []
        event_records = []  # per-event dict, for the event-display figure (pretrained only)
        with torch.no_grad():
            for i, emb, lab, feat in zip(test_idx, test_embs, test_labels, test_feats):
                out = probe(torch.from_numpy(emb).float().cuda()).cpu().numpy()
                res = cluster_and_match(out, lab, args.seed, args.double_majority_threshold)
                if res is None:
                    continue
                aris.append(res["ari"]); fracs_found.append(res["frac_found"]); hit_fracs.append(res["hit_frac_found"])
                event_records.append(dict(event=i, feat=feat, true_labels=lab, **res))

        aris, fracs_found, hit_fracs = np.array(aris), np.array(fracs_found), np.array(hit_fracs)
        results[label] = dict(ari=aris, frac_found=fracs_found, hit_frac=hit_fracs)
        print(f"[RESULT] {label}: mean ARI = {aris.mean():.4f} +/- {aris.std():.4f} | "
              f"mean fraction of tracks found (double-majority) = {fracs_found.mean():.4f} +/- {fracs_found.std():.4f} | "
              f"mean fraction of hits in a found track = {hit_fracs.mean():.4f} +/- {hit_fracs.std():.4f} "
              f"(n={len(aris)} valid events)")

        if label == "pretrained":
            plot_material = event_records

    p, r = results["pretrained"], results["random_init"]
    n = min(len(p["ari"]), len(r["ari"]))
    ari_delta = p["ari"][:n] - r["ari"][:n]
    found_delta = p["frac_found"][:n] - r["frac_found"][:n]

    print("\n=== SUMMARY (track-finding evaluation, held-out test events) ===")
    print(f"ARI              : pretrained {p['ari'].mean():.4f} vs random-init {r['ari'].mean():.4f} "
          f"(mean delta {ari_delta.mean():+.4f}, n={n})")
    print(f"Frac. tracks found: pretrained {p['frac_found'].mean():.4f} vs random-init {r['frac_found'].mean():.4f} "
          f"(mean delta {found_delta.mean():+.4f}, n={n})")
    if n >= 6:
        _, p_ari = sstats.wilcoxon(ari_delta)
        _, p_found = sstats.wilcoxon(found_delta) if np.any(found_delta != 0) else (None, float("nan"))
        print(f"Wilcoxon p (ARI delta): {p_ari:.4f}")
        print(f"Wilcoxon p (frac-found delta): {p_found:.4f}")
    else:
        print("[INFO] fewer than 6 paired events -- skipping significance test.")

    # --- summary figure: ARI and track-finding-efficiency, pretrained vs random-init ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].boxplot([p["ari"][:n], r["ari"][:n]], tick_labels=["pretrained", "random-init"])
    axes[0].set_ylabel("Adjusted Rand Index (per event)")
    axes[0].set_title(f"Clustering-vs-truth agreement\n(n={n}, KMeans k=n_true_tracks)")
    axes[1].boxplot([p["frac_found"][:n], r["frac_found"][:n]], tick_labels=["pretrained", "random-init"])
    axes[1].set_ylabel("fraction of tracks found (double-majority)")
    axes[1].set_title(f"Track-finding efficiency\n(threshold={args.double_majority_threshold})")
    fig.tight_layout()
    fig.savefig(args.summary_out, dpi=150)
    print(f"[DONE] Saved summary figure to {args.summary_out}")

    # --- event-display figure: generated (blue) vs reconstructed (red x) hits, real coordinates ---
    candidates = [r_ for r_ in plot_material if len(np.unique(r_["true_labels"])) >= 2]
    candidates.sort(key=lambda r_: r_["feat"].shape[0], reverse=True)
    chosen = candidates[:args.n_plot_events]

    ncols = min(3, max(1, len(chosen)))
    nrows = int(np.ceil(len(chosen) / ncols)) if chosen else 1
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
    for idx, rec in enumerate(chosen):
        ax = axes2[idx // ncols][idx % ncols]
        feat, true_labels, correct_mask = rec["feat"], rec["true_labels"], rec["correct_mask"]
        if args.plot_coords == "xy":
            cx, cy = feat[:, 1], feat[:, 2]
            xlabel, ylabel = "x (cm)", "y (cm)"
        else:
            eta, phi, _ = cartesian_to_polar(feat[:, 1], feat[:, 2], feat[:, 3])
            cx, cy = eta, phi
            xlabel, ylabel = "eta", "phi"
        ax.scatter(cx, cy, c="tab:blue", s=18, label="generated hits", zorder=1)
        ax.scatter(cx[correct_mask], cy[correct_mask], marker="x", c="red", s=45,
                   linewidths=1.5, label="reconstructed hits", zorder=2)
        ax.set_title(f"event {rec['event']} | {feat.shape[0]} hits, {len(np.unique(true_labels))} tracks\n"
                     f"ARI={rec['ari']:.2f}, tracks found={rec['frac_found']:.0%}")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if idx == 0:
            ax.legend(fontsize=8, loc="best")
    for idx in range(len(chosen), nrows * ncols):
        axes2[idx // ncols][idx % ncols].axis("off")
    fig2.suptitle("Pretrained backbone: generated vs. reconstructed hits "
                   "(double-majority-matched tracks only)")
    fig2.tight_layout()
    fig2.savefig(args.events_out, dpi=150)
    print(f"[DONE] Saved event-display figure to {args.events_out}")


if __name__ == "__main__":
    main()
