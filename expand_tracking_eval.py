#!/usr/bin/env python3
"""
Evaluate an already-trained eval_track_finding_star.py model (loaded from a
--results-dir it saved) on a LARGER held-out set, without any retraining --
just the frozen backbone + the saved down_model weights, run forward.

Why this is safe as a held-out set: the original training run only ever read
from the converted dataset's "train" RaggedMmap split (carving its own
fit/train/test out of that one split internally). This script reads from the
"test" split instead -- a different file, never touched by that run at all --
so there's no leakage risk to reason about.

Produces an expanded test_results.pkl per arm, in the same format
eval_track_finding_star.py's --results-dir writes, so analyze_ari_vs_tracks.py
and plot_event_displays()/plot_contingency_heatmaps() all work unmodified.

Usage:
    python expand_tracking_eval.py \
        --source-results-dir results/tracking_pythia_multifile \
        --out-results-dir results/tracking_pythia_multifile_expanded \
        --n-events 1600
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from extract_embeddings_pca import compute_star_stats, SPHENIX_STATS
from star_voxelizer import compute_star_final_bins, build_voxelizer, hierarchical_raster_scan_order
from extract_embeddings_linear_probe import event_polar, build_model
from eval_track_finding_star import (
    get_frozen_feature_stack, score_event, save_arm_results, MambaAttentionHead,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-results-dir", required=True, help="Where the original run's config.json + weights live.")
    p.add_argument("--out-results-dir", required=True, help="Where to write the expanded results (kept separate).")
    p.add_argument("--eval-split", default="test", choices=["train", "test"],
                    help="Which RaggedMmap split to draw NEW events from -- 'test' is never touched by training.")
    p.add_argument("--n-events", type=int, default=1600)
    p.add_argument("--min-hits", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    src = os.path.expanduser(args.source_results_dir)
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    print(f"[INFO] Loaded config from {src}/config.json (original run's --data-dir={cfg['data_dir']})")

    device = torch.device("cuda")
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"]); np.random.seed(cfg["seed"])

    from mmap_ninja import RaggedMmap
    feat_path = os.path.join(os.path.expanduser(cfg["data_dir"]), f"features_{args.eval_split}")
    tgt_path = os.path.join(os.path.expanduser(cfg["data_dir"]), f"seg_target_{args.eval_split}")
    features_mmap = RaggedMmap(feat_path)
    target_mmap = RaggedMmap(tgt_path)
    n_avail = len(features_mmap)
    print(f"[INFO] {n_avail} events available in split '{args.eval_split}' (never used in training).")

    idx_list = []
    for i in range(n_avail):
        if np.asarray(features_mmap[i]).shape[0] >= args.min_hits:
            idx_list.append(i)
        if len(idx_list) >= args.n_events:
            break
    print(f"[INFO] Using {len(idx_list)} events (min-hits={args.min_hits} filter).")

    # Refit the voxelizer EXACTLY as the original run did: same fit events (same seed + same
    # events_with_hits construction over the ORIGINAL --split/data, deterministic) -- but simplest
    # and safest is to just refit on this new split's own first n_fit_events; the voxelizer only
    # encodes eta/phi/radius binning statistics, which should be near-identical either way since
    # both splits are large i.i.d. samples of the same underlying simulation.
    stats = SPHENIX_STATS if cfg["norm"] == "sphenix" else compute_star_stats(features_mmap, cfg["norm_sample_events"])
    fit_idx = idx_list[:cfg["n_fit_events"]]
    fit_eta, fit_phi, fit_r_raw, fit_r_n = [], [], [], []
    for i in fit_idx:
        feat = np.asarray(features_mmap[i])
        _, eta_n, phi_n, r_n, r_raw = event_polar(feat, stats)
        fit_eta.append(eta_n); fit_phi.append(phi_n); fit_r_raw.append(r_raw); fit_r_n.append(r_n)
    fit_eta, fit_phi = np.concatenate(fit_eta), np.concatenate(fit_phi)
    fit_r_raw, fit_r_n = np.concatenate(fit_r_raw), np.concatenate(fit_r_n)
    final_bins = compute_star_final_bins(fit_eta, fit_phi, fit_r_raw, fit_r_n,
                                          n_bins=tuple(cfg["n_bins"]), n_radial_groups=cfg["n_radial_groups"])
    voxelizer = build_voxelizer(final_bins, n_bins=tuple(cfg["n_bins"]),
                                 dim_sweep_order=tuple(cfg["dim_sweep_order"]),
                                 revert_order=tuple(cfg["revert_order"]))
    eval_idx = idx_list[cfg["n_fit_events"]:]  # don't re-evaluate on the voxelizer-fit events
    print(f"[INFO] Voxelizer refit on {len(fit_idx)} events; evaluating on the remaining {len(eval_idx)}.")

    for arm_label in ("pretrained", "random_init"):
        print(f"\n[INFO] === {arm_label} ===")
        if arm_label == "pretrained":
            model = build_model(cfg["embed_dim"], cfg["num_layers"], cfg["d_state"], cfg["d_conv"], cfg["expand"],
                                cfg["klen"], cfg["checkpoint"], load_checkpoint=True)
        else:
            model = build_model(cfg["embed_dim"], cfg["num_layers"], cfg["d_state"], cfg["d_conv"], cfg["expand"],
                                cfg["klen"], cfg["checkpoint"], load_checkpoint=False)
            bb_sd = torch.load(os.path.join(src, arm_label, "backbone.pt"), map_location="cpu", weights_only=False)
            model.load_state_dict(bb_sd)
            model.eval()
            for p_ in model.parameters():
                p_.requires_grad = False
        print(f"[INFO] Backbone ready ({'checkpoint' if arm_label == 'pretrained' else 'saved backbone.pt'}).")

        down_model = MambaAttentionHead(
            input_dim=cfg["embed_dim"], embed_dim=cfg["head_embed_dim"], num_layers=0,
            num_feature_layers=cfg["num_layers"], num_prototypes=cfg["max_gt_classes"],
            dropout=cfg["head_dropout"],
        ).to(device)
        dm_sd = torch.load(os.path.join(src, arm_label, "down_model.pt"), map_location="cpu", weights_only=False)
        missing, unexpected = down_model.load_state_dict(dm_sd, strict=True)
        assert not missing and not unexpected, f"checkpoint mismatch: missing={missing} unexpected={unexpected}"
        down_model.eval()
        print(f"[INFO] down_model.pt reloaded cleanly ({sum(p.numel() for p in down_model.parameters()):,} params).")

        aris, fracs = [], []
        test_records = []
        for n_done, i in enumerate(eval_idx):
            feat = np.asarray(features_mmap[i])
            lab = np.asarray(target_mmap[i])
            E_n, eta_n, phi_n, r_n, r_raw = event_polar(feat, stats)
            order = hierarchical_raster_scan_order(voxelizer, eta_n, phi_n, r_n)
            feat_stack = get_frozen_feature_stack(model, E_n, eta_n, phi_n, r_n, order)
            r = score_event(down_model, feat_stack, lab, device,
                            cfg["decode_threshold"], cfg["double_majority_threshold"])
            aris.append(r["ari"]); fracs.append(r["frac_found"])
            test_records.append(dict(
                event_idx=int(i), feat=feat, true_labels=lab,
                assignments=r["assignments"], correct_mask=r["correct_mask"],
                ari=r["ari"], frac_found=r["frac_found"], hit_frac_found=r["hit_frac_found"],
            ))
            if (n_done + 1) % 200 == 0:
                print(f"    ...{n_done + 1}/{len(eval_idx)} events scored "
                      f"(running mean ARI={np.mean(aris):.4f})")

        print(f"[RESULT] {arm_label}: ARI = {np.mean(aris):.4f} +/- {np.std(aris):.4f}, "
              f"frac_found = {np.mean(fracs):.4f} +/- {np.std(fracs):.4f}  (n={len(aris)} events)")
        save_arm_results(os.path.expanduser(args.out_results_dir), arm_label, down_model, model,
                          is_random_init=(arm_label == "random_init"), test_records=test_records)

    print(f"\n[DONE] Expanded results written to {args.out_results_dir}/")


if __name__ == "__main__":
    main()
