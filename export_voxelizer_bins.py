#!/usr/bin/env python3
"""
DRAFT companion to configs/star_tracking.yaml -- not yet wired into a real
training run.

star_voxelizer.py fits STAR-appropriate eta/phi/radius bins and then
constructs a working `Voxelizer` by bypassing its normal __init__ (which
expects to load bin edges from a pickled file on disk, or compute them from a
live TPCBatchDataset + DataLoader). That bypass is fine for this repo's own
probe scripts, which build their own Voxelizer object directly -- but FM4NPP's
REAL `fm4npp.datasets.dataset.TPCBatchDataset` (used by
train/downstream/track_finding_trainer.py) always goes through the normal,
un-bypassed `Voxelizer.__init__`, which looks for a file at:

    <stat_dir>/bin_edges_<bin_version>_nbins_<n0>_<n1>_<n2>.pkl

(fm4npp/datasets/voxelizer.py, Voxelizer.__init__ / pickle_load / pickle_save
 -- confirmed by reading the actual code: pickle_save/pickle_load just do a
 plain `pickle.dump(self.final_bins, f)` / `pickle.load(f)`, no special
 format, so our own `final_bins` dict drops in directly.)

This script fits those bins the same way extract_embeddings_linear_probe.py
does (same --n-fit-events events, same event_polar/compute_star_final_bins
calls) and writes the pickle to exactly that path, so the REAL, unmodified
TPCBatchDataset/Voxelizer can load STAR-fitted bins transparently -- no
FM4NPP code needs to be patched for this part.

Usage:
    python export_voxelizer_bins.py \
        --data-dir data/star_fm4npp_2k \
        --stat-dir stats \
        --n-fit-events 60 --norm star \
        --bin-version v3 --n-bins 8 8 6
"""

import argparse
import os
import pickle

import numpy as np

from extract_embeddings_pca import compute_star_stats, SPHENIX_STATS
from extract_embeddings_linear_probe import event_polar
from star_voxelizer import compute_star_final_bins


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--stat-dir", required=True, help="Output directory -- must match configs/*.yaml's stat_dir.")
    p.add_argument("--n-fit-events", type=int, default=60)
    p.add_argument("--min-hits", type=int, default=15)
    p.add_argument("--norm", default="star", choices=["star", "sphenix"])
    p.add_argument("--norm-sample-events", type=int, default=200)
    p.add_argument("--bin-version", default="v3", help="Must match Voxelizer's bin_version arg (FM4NPP uses 'v3').")
    p.add_argument("--n-bins", type=int, nargs=3, default=[8, 8, 6], metavar=("ETA", "PHI", "RADIUS"))
    p.add_argument("--n-radial-groups", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    from mmap_ninja import RaggedMmap

    feat_path = os.path.join(os.path.expanduser(args.data_dir), f"features_{args.split}")
    features_mmap = RaggedMmap(feat_path)

    fit_idx = []
    for i in range(len(features_mmap)):
        if np.asarray(features_mmap[i]).shape[0] >= args.min_hits:
            fit_idx.append(i)
        if len(fit_idx) >= args.n_fit_events:
            break
    if len(fit_idx) < args.n_fit_events:
        raise SystemExit(f"[ERROR] only {len(fit_idx)} events pass --min-hits, need {args.n_fit_events}.")
    print(f"[INFO] Fitting bins on {len(fit_idx)} events.")

    stats = SPHENIX_STATS if args.norm == "sphenix" else compute_star_stats(features_mmap, args.norm_sample_events)

    fit_eta, fit_phi, fit_r_raw, fit_r_n = [], [], [], []
    for i in fit_idx:
        feat = np.asarray(features_mmap[i])
        _, eta_n, phi_n, r_n, r_raw = event_polar(feat, stats)
        fit_eta.append(eta_n); fit_phi.append(phi_n); fit_r_raw.append(r_raw); fit_r_n.append(r_n)
    fit_eta, fit_phi = np.concatenate(fit_eta), np.concatenate(fit_phi)
    fit_r_raw, fit_r_n = np.concatenate(fit_r_raw), np.concatenate(fit_r_n)

    final_bins = compute_star_final_bins(fit_eta, fit_phi, fit_r_raw, fit_r_n,
                                          n_bins=tuple(args.n_bins), n_radial_groups=args.n_radial_groups)

    os.makedirs(os.path.expanduser(args.stat_dir), exist_ok=True)
    bin_addr = os.path.join(os.path.expanduser(args.stat_dir),
                             "bin_edges_{}_nbins_{}_{}_{}.pkl".format(args.bin_version, *args.n_bins))
    with open(bin_addr, "wb") as f:
        pickle.dump(final_bins, f)
    print(f"[DONE] Wrote {bin_addr}")
    print(f"       keys: {list(final_bins.keys())}, "
          f"lengths: {[len(v) for v in final_bins.values()]}")
    print("       NOTE: this only sets the bin EDGES. dim_sweep_order/revert_order "
          "(order: EPR in configs/star_tracking.yaml) are a separate Voxelizer "
          "constructor argument, read from the training config, not from this file.")


if __name__ == "__main__":
    main()
