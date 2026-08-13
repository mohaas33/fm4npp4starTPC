#!/usr/bin/env python3
"""
Load the pretrained FM4NPP checkpoint (frozen, no training) and extract
per-hit embeddings for a handful of converted STAR events, then visualize
with PCA and compute a silhouette score against true track-id labels.

This replicates the REAL preprocessing pipeline used in
fm4npp/datasets/dataset_pretrain.py's __getitem__:
    (E, x, y, z) -> polar (E, eta, phi, r) -> normalize -> sort by r
It intentionally skips the Voxelizer/space-filling-curve serialization step,
since that requires bin-stat .pkl files not included in the public repo.
Plain r-sorted normalized points is a genuine code path in the original
dataset class (the 'else' branch when voxelize=False), so this is a faithful
subset of their real pipeline, not an invented substitute.

Two normalization modes (see original planning discussion):
    --norm sphenix  : use the pretrained model's own hardcoded sPHENIX
                      normalization constants unchanged (true zero-shot test)
    --norm star     : recompute eta/phi/r/E normalization stats from your
                      own STAR sample (isolates representation transfer from
                      unit/scale mismatch)

Usage:
    python extract_embeddings_pca.py \
        --data-dir ~/Work/STAR/data/star_fm4npp_2k \
        --checkpoint ~/Work/STAR/checkpoints/pp_nerf_m3_k30.ckpt \
        --split train \
        --n-events 6 \
        --min-hits 15 \
        --norm star \
        --out embeddings_pca.png

Requires: torch, mmap_ninja, scikit-learn, matplotlib (in addition to the
FM4NPP env you already have working: mamba-ssm, causal-conv1d, etc.)
    pip install scikit-learn matplotlib
"""

import argparse
import os
import sys

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="Output dir from convert_star_to_fm4npp.py")
    p.add_argument("--checkpoint", required=True, help="Path to pp_nerf_m3_k30.ckpt (or similar)")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--n-events", type=int, default=6, help="How many events to compute statistics over")
    p.add_argument("--n-plot-events", type=int, default=6,
                    help="How many INDIVIDUAL events to show as PCA scatter panels (kept small on purpose -- "
                         "readability degrades fast past ~8. Statistics still use all --n-events events "
                         "regardless of this setting.)")
    p.add_argument("--min-hits", type=int, default=15,
                    help="Only pick events with at least this many hits (tiny events give uninformative plots)")
    p.add_argument("--norm", default="star", choices=["star", "sphenix"],
                    help="'star' recomputes normalization stats from your data; "
                         "'sphenix' keeps the pretrained model's original hardcoded constants (true zero-shot)")
    p.add_argument("--norm-sample-events", type=int, default=200,
                    help="How many events to scan when recomputing STAR normalization stats (--norm star only)")
    p.add_argument("--out", default="embeddings_pca.png",
                    help="Filename for the small per-event PCA panel figure (--n-plot-events events)")
    p.add_argument("--summary-out", default="silhouette_summary.png",
                    help="Filename for the aggregate summary figure (box plot + delta histogram, all --n-events events)")
    p.add_argument("--compare-random", action="store_true",
                    help="Also build a second, RANDOMLY INITIALIZED model of the same architecture "
                         "(checkpoint weights NOT loaded) and report its silhouette scores on the "
                         "exact same events, side by side with the pretrained model. This is the "
                         "control that tells you whether the pretrained weights are doing anything, "
                         "vs. any model of this shape showing some structure just from raw geometry.")

    # Model architecture -- must match the checkpoint exactly (derived earlier from its state_dict shapes)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--d-state", type=int, default=32)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--klen", type=int, default=30)
    return p.parse_args()


# ---- polar conversion + normalization, matching fm4npp/utils.py + dataset_pretrain.py exactly ----

def cartesian_to_polar(x, y, z):
    """x,y,z: (N,) numpy arrays. Returns eta, phi, r (transverse)."""
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    eta = np.arctanh(np.clip(z / np.sqrt(x**2 + y**2 + z**2), -1 + 1e-7, 1 - 1e-7))
    return eta, phi, r


def znormalize(arr, mean_, std_):
    return (arr - mean_) / std_


def minmax_normalize(arr, max_, min_):
    return (arr - min_) / (max_ - min_)


SPHENIX_STATS = dict(
    eta_lim=(-2.0, 2.0),
    phi_lim=(-np.pi, np.pi),
    r_lim=(31.371997833251953, 75.38493347167969),
    E_mean=253.0982,
    E_std=268.7093,
)


def polar_and_normalize(feat_exyz, stats):
    """feat_exyz: (N,4) numpy array = (E,x,y,z). Returns (N,4) normalized (E,eta,phi,r), r-sort index."""
    E = feat_exyz[:, 0]
    x, y, z = feat_exyz[:, 1], feat_exyz[:, 2], feat_exyz[:, 3]
    eta, phi, r = cartesian_to_polar(x, y, z)

    E_n = znormalize(E, stats["E_mean"], stats["E_std"])
    eta_n = minmax_normalize(eta, stats["eta_lim"][1], stats["eta_lim"][0])
    phi_n = minmax_normalize(phi, stats["phi_lim"][1], stats["phi_lim"][0])
    r_n = minmax_normalize(r, stats["r_lim"][1], stats["r_lim"][0])

    out = np.stack([E_n, eta_n, phi_n, r_n], axis=-1).astype(np.float32)
    sort_idx = np.argsort(r)  # sort by RAW r, matching dataset_pretrain.py (sorts before normalizing r further)
    return out[sort_idx], sort_idx


def compute_star_stats(features_mmap, n_scan):
    """Recompute eta/phi/r/E normalization stats from a sample of STAR events."""
    all_eta, all_phi, all_r, all_E = [], [], [], []
    n_scan = min(n_scan, len(features_mmap))
    for i in range(n_scan):
        feat = np.asarray(features_mmap[i])
        if feat.shape[0] == 0:
            continue
        E, x, y, z = feat[:, 0], feat[:, 1], feat[:, 2], feat[:, 3]
        eta, phi, r = cartesian_to_polar(x, y, z)
        all_eta.append(eta); all_phi.append(phi); all_r.append(r); all_E.append(E)
    all_eta = np.concatenate(all_eta); all_phi = np.concatenate(all_phi)
    all_r = np.concatenate(all_r); all_E = np.concatenate(all_E)
    stats = dict(
        eta_lim=(float(all_eta.min()), float(all_eta.max())),
        phi_lim=(float(all_phi.min()), float(all_phi.max())),
        r_lim=(float(all_r.min()), float(all_r.max())),
        E_mean=float(all_E.mean()),
        E_std=float(all_E.std() + 1e-8),
    )
    print(f"[INFO] Recomputed STAR stats from {n_scan} events:")
    for k, v in stats.items():
        print(f"       {k} = {v}")
    return stats


def main():
    args = parse_args()

    try:
        from mmap_ninja import RaggedMmap
    except ImportError:
        sys.exit("mmap-ninja is required: pip install mmap-ninja")
    try:
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
    except ImportError:
        sys.exit("scikit-learn is required: pip install scikit-learn")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")

    from fm4npp.models.mambagpt import MambaGPT

    data_dir = os.path.expanduser(args.data_dir)
    feat_path = os.path.join(data_dir, f"features_{args.split}")
    tgt_path = os.path.join(data_dir, f"seg_target_{args.split}")
    print(f"[INFO] Loading RaggedMmap from {feat_path} / {tgt_path}")
    features_mmap = RaggedMmap(feat_path)
    target_mmap = RaggedMmap(tgt_path)
    print(f"[INFO] {len(features_mmap)} events available in split '{args.split}'")

    # --- normalization stats ---
    if args.norm == "sphenix":
        stats = SPHENIX_STATS
        print("[INFO] Using ORIGINAL sPHENIX normalization constants (true zero-shot test).")
    else:
        stats = compute_star_stats(features_mmap, args.norm_sample_events)

    def build_model(load_checkpoint):
        m = MambaGPT(embed_dim=args.embed_dim, num_layers=args.num_layers, d_state=args.d_state,
                     d_conv=args.d_conv, expand=args.expand, klen=args.klen,
                     dropout=0.0, embed_method='add', pe_method='nerf').cuda()
        if load_checkpoint:
            ckpt = torch.load(os.path.expanduser(args.checkpoint), map_location='cpu', weights_only=False)
            state_dict = {k.replace('module.', '', 1): v for k, v in ckpt['model_state'].items()}
            missing, unexpected = m.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[WARN] missing={missing} unexpected={unexpected}")
            else:
                print("[INFO] Checkpoint loaded cleanly (0 missing, 0 unexpected keys).")
        else:
            print("[INFO] Using RANDOM initialization (checkpoint NOT loaded) -- this is the control model.")
        m.eval()
        for p_ in m.parameters():
            p_.requires_grad = False  # frozen either way -- no training happens in this script
        return m

    print("[INFO] Building pretrained model (frozen, eval mode)...")
    model_pretrained = build_model(load_checkpoint=True)

    model_random = None
    if args.compare_random:
        print("[INFO] Building random-init control model (frozen, eval mode)...")
        model_random = build_model(load_checkpoint=False)

    # --- pick events with enough hits ---
    chosen = []
    for i in range(len(features_mmap)):
        feat = np.asarray(features_mmap[i])
        if feat.shape[0] >= args.min_hits:
            chosen.append(i)
        if len(chosen) >= args.n_events:
            break
    if not chosen:
        sys.exit(f"[ERROR] No events with >= {args.min_hits} hits found. Try lowering --min-hits.")
    print(f"[INFO] Using {len(chosen)} events: {chosen}")

    def get_embedding_and_silhouette(model, norm_feat, track_id_sorted):
        x_in = torch.from_numpy(norm_feat).unsqueeze(0).cuda()  # (1, N, 4) -- channel-LAST, matches EmbedderAdd
        with torch.no_grad():
            _, feature_layers, _ = model(x_in, return_z=True)
        emb = feature_layers[-1].squeeze(0).cpu().numpy()  # last mamba layer's per-point output
        n_unique = len(np.unique(track_id_sorted))
        if n_unique >= 2 and emb.shape[0] > n_unique:
            sil = silhouette_score(emb, track_id_sorted)
        else:
            sil = float("nan")
        return emb, sil

    # --- extract embeddings/silhouette for ALL chosen events (stats), plot only first --n-plot-events (readability) ---
    n_plot = min(args.n_plot_events, len(chosen))
    n_rows = 2 if args.compare_random else 1
    fig, axes = plt.subplots(n_rows, n_plot, figsize=(4.2 * n_plot, 4.2 * n_rows), squeeze=False)

    sil_pretrained_all, sil_random_all, hits_all, ntrack_all = [], [], [], []

    for idx, ev_idx in enumerate(chosen):
        feat_exyz = np.asarray(features_mmap[ev_idx])   # (N,4) = E,x,y,z
        track_id = np.asarray(target_mmap[ev_idx])        # (N,)
        norm_feat, sort_idx = polar_and_normalize(feat_exyz, stats)
        track_id_sorted = track_id[sort_idx]
        n_unique_tracks = len(np.unique(track_id_sorted))
        hits_all.append(feat_exyz.shape[0])
        ntrack_all.append(n_unique_tracks)

        emb_p, sil_p = get_embedding_and_silhouette(model_pretrained, norm_feat, track_id_sorted)
        sil_pretrained_all.append(sil_p)

        line = (f"[RESULT] event {ev_idx}: {feat_exyz.shape[0]} hits, {n_unique_tracks} tracks | "
                f"pretrained silhouette={sil_p:.4f}")

        sil_r = None
        if args.compare_random:
            emb_r, sil_r = get_embedding_and_silhouette(model_random, norm_feat, track_id_sorted)
            sil_random_all.append(sil_r)
            line += f" | random-init silhouette={sil_r:.4f} | delta={sil_p - sil_r:+.4f}"

        print(line)

        # only render a PCA panel for the first n_plot events -- keeps the figure legible
        if idx < n_plot:
            emb2d_p = PCA(n_components=2).fit_transform(emb_p)
            ax = axes[0, idx]
            ax.scatter(emb2d_p[:, 0], emb2d_p[:, 1], c=track_id_sorted, cmap="tab20", s=25)
            ax.set_title(f"event {ev_idx} | {feat_exyz.shape[0]} hits | {n_unique_tracks} tracks\n"
                         f"PRETRAINED sil={sil_p:.3f}", fontsize=9)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            if args.compare_random:
                emb2d_r = PCA(n_components=2).fit_transform(emb_r)
                ax2 = axes[1, idx]
                ax2.scatter(emb2d_r[:, 0], emb2d_r[:, 1], c=track_id_sorted, cmap="tab20", s=25)
                ax2.set_title(f"RANDOM-INIT sil={sil_r:.3f}", fontsize=9)
                ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")

    fig.suptitle(f"FM4NPP embeddings, PCA-projected, colored by true track id (norm={args.norm})\n"
                 f"showing {n_plot} of {len(chosen)} total events used for statistics"
                 + ("  [top: pretrained, bottom: random-init]" if args.compare_random else ""),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[DONE] Saved per-event panel figure ({n_plot} events) to {args.out}")

    print()
    if args.compare_random:
        p_valid = np.array([s for s in sil_pretrained_all if not np.isnan(s)])
        r_valid = np.array([s for s, sp in zip(sil_random_all, sil_pretrained_all) if not np.isnan(sp) and not np.isnan(s)])
        p_paired = np.array([sp for sp, sr in zip(sil_pretrained_all, sil_random_all) if not np.isnan(sp) and not np.isnan(sr)])
        deltas = p_paired - r_valid

        print("=== SUMMARY (all {} events) ===".format(len(chosen)))
        print(f"Mean silhouette, pretrained : {p_valid.mean():.4f}  (std={p_valid.std():.4f}, n={len(p_valid)})")
        print(f"Mean silhouette, random-init: {r_valid.mean():.4f}  (std={r_valid.std():.4f}, n={len(r_valid)})")
        print(f"Mean delta (pretrained - random): {deltas.mean():+.4f}  (std={deltas.std():.4f}, n={len(deltas)})")

        # paired statistical test on the deltas -- is the pretrained-vs-random gap real, or noise?
        try:
            from scipy import stats as sstats
            if len(deltas) >= 6:
                w_stat, w_p = sstats.wilcoxon(deltas)
                t_stat, t_p = sstats.ttest_1samp(deltas, 0.0)
                print(f"Wilcoxon signed-rank test on deltas: p={w_p:.4f}")
                print(f"Paired t-test on deltas:             p={t_p:.4f}")
                if w_p < 0.05:
                    print(">>> p < 0.05: the pretrained-vs-random gap is unlikely to be pure noise.")
                else:
                    print(">>> p >= 0.05: cannot rule out that the gap is just noise at this sample size.")
            else:
                print("[INFO] Fewer than 6 valid events -- skipping statistical test (not enough for it to mean anything).")
        except ImportError:
            print("[INFO] scipy not available, skipping statistical test. pip install scipy for this.")

        # --- aggregate summary figure: box plot + delta histogram, ALL events at once ---
        fig2, (axb, axh) = plt.subplots(1, 2, figsize=(11, 4.5))
        axb.boxplot([p_valid, r_valid], tick_labels=["pretrained", "random-init"])
        axb.set_ylabel("silhouette score")
        axb.set_title(f"Silhouette score distribution\n({len(chosen)} events, norm={args.norm})")

        axh.hist(deltas, bins=min(20, max(5, len(deltas) // 3)), color="steelblue", edgecolor="black")
        axh.axvline(0, color="red", linestyle="--", label="zero (no effect)")
        axh.axvline(deltas.mean(), color="black", linestyle="-", label=f"mean = {deltas.mean():+.4f}")
        axh.set_xlabel("delta (pretrained - random-init), per event")
        axh.set_ylabel("count")
        axh.set_title("Per-event delta distribution")
        axh.legend(fontsize=8)

        fig2.tight_layout()
        fig2.savefig(args.summary_out, dpi=150)
        print(f"[DONE] Saved aggregate summary figure to {args.summary_out}")

        print()
        print("How to read this: the boxplot shows overall spread of raw scores; the histogram is the")
        print("more decisive one -- it's the per-event (pretrained - random) gap. If that distribution")
        print("sits clearly to the right of zero (and the statistical test agrees), that's real evidence")
        print("the pretrained weights transfer something. If it straddles zero, the earlier positive")
        print("silhouette scores were mostly an artifact of feeding spatially-correlated coordinates")
        print("into any model of this shape, pretrained or not.")
    else:
        print("Re-run with --compare-random to see whether these silhouette scores actually reflect")
        print("something the pretrained weights learned, versus what any same-shaped random model")
        print("would show on this data. Without that comparison, this number alone isn't conclusive.")


if __name__ == "__main__":
    main()