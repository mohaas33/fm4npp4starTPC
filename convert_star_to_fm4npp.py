#!/usr/bin/env python3
"""
Convert a STAR TPC simulation ROOT TTree into the RaggedMmap format expected
by FM4NPP's TPCBatchDataset (fm4npp/datasets/dataset_pretrain.py, dataset.py).

Each event's hits become:
    features   : (N_hits, 4) float32   -> columns (E, x, y, z)
                 E is taken from TPCHits_adc (raw ADC ionization signal),
                 matching the sPHENIX "E" feature definition the pretrained
                 model was trained on. x,y,z are used as-is (cm, Cartesian).
    seg_target : (N_hits,)   int64     -> TPCHits_IdTruth (track id per hit)
    reg_target : (N_hits, 7) float32   -> ALL ZEROS. TPCBatchDataset
                 unconditionally loads a reg_target RaggedMmap even though
                 it's only used for an auxiliary regression loss during
                 downstream training, not for embedding extraction. The STAR
                 hit tree in this project has no truth momentum/vertex info
                 per hit, so this is a placeholder to satisfy the loader.
                 Real reg_target values matter only if/when you train with
                 return_reg=True; for the frozen-backbone probe they are
                 never read.

Output layout (mmap_ninja RaggedMmap, matches SETUP.md / dataset_pretrain.py):
    <output_dir>/
        features_train/     seg_target_train/     reg_target_train/
        features_test/      seg_target_test/      reg_target_test/

Usage:
    python convert_star_to_fm4npp.py \
        --input /path/to/star_sim.root \
        --tree T \
        --output-dir ~/Work/STAR/data/star_fm4npp \
        --max-events 2000 \
        --train-frac 0.8

    # Convert everything in the file (no limit):
    python convert_star_to_fm4npp.py --input file.root --tree T --output-dir out/

Requires: uproot, awkward, numpy, mmap_ninja
    pip install uproot awkward mmap-ninja
"""

import argparse
import os
import shutil
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Path to input ROOT file")
    p.add_argument("--tree", required=True, help="Name of the TTree inside the ROOT file (e.g. 'T')")
    p.add_argument("--output-dir", required=True, help="Directory to write RaggedMmap output into")
    p.add_argument("--max-events", type=int, default=None,
                    help="Only convert the first N events (for quick tests). Default: convert all events.")
    p.add_argument("--train-frac", type=float, default=0.8,
                    help="Fraction of converted events assigned to the 'train' split (rest go to 'test'). Default 0.8")
    p.add_argument("--seed", type=int, default=42, help="Random seed for the train/test split shuffle")
    p.add_argument("--min-hits", type=int, default=1,
                    help="Skip events with fewer than this many hits (default 1, i.e. keep everything with >=1 hit)")
    p.add_argument("--x-branch", default="TPCHits_x")
    p.add_argument("--y-branch", default="TPCHits_y")
    p.add_argument("--z-branch", default="TPCHits_z")
    p.add_argument("--e-branch", default="TPCHits_adc",
                    help="Branch used for the energy/'E' feature. Default TPCHits_adc "
                         "(matches sPHENIX's raw ADC ionization signal used for E in the pretrained model). "
                         "Use TPCHits_q instead if you want calibrated charge; note this changes the "
                         "value scale the pretrained model's normalization constants were tuned for.")
    p.add_argument("--track-branch", default="TPCHits_IdTruth", help="Branch used for the track-id target")
    p.add_argument("--overwrite", action="store_true", help="Delete output-dir first if it already exists")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import uproot
    except ImportError:
        sys.exit("uproot is required: pip install uproot awkward")
    try:
        from mmap_ninja import RaggedMmap
    except ImportError:
        sys.exit("mmap-ninja is required: pip install mmap-ninja")

    out_dir = os.path.expanduser(args.output_dir)
    if os.path.exists(out_dir):
        if args.overwrite:
            print(f"[INFO] --overwrite set, removing existing {out_dir}")
            shutil.rmtree(out_dir)
        else:
            sys.exit(f"[ERROR] {out_dir} already exists. Use --overwrite to replace it, "
                      f"or pick a different --output-dir.")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Opening {args.input} : {args.tree}")
    tree = uproot.open(args.input)[args.tree]

    n_available = tree.num_entries
    n_events = n_available if args.max_events is None else min(args.max_events, n_available)
    print(f"[INFO] Tree has {n_available} events; converting {n_events}"
          + (f" (limited by --max-events)" if args.max_events is not None else ""))

    branches = [args.x_branch, args.y_branch, args.z_branch, args.e_branch, args.track_branch]
    arrays = tree.arrays(branches, entry_stop=n_events, library="np")

    x_all = arrays[args.x_branch]
    y_all = arrays[args.y_branch]
    z_all = arrays[args.z_branch]
    e_all = arrays[args.e_branch]
    tid_all = arrays[args.track_branch]

    features_list = []
    seg_target_list = []
    reg_target_list = []
    kept_hit_counts = []
    skipped_short = 0

    for i in range(n_events):
        x = np.asarray(x_all[i], dtype=np.float32)
        y = np.asarray(y_all[i], dtype=np.float32)
        z = np.asarray(z_all[i], dtype=np.float32)
        e = np.asarray(e_all[i], dtype=np.float32)
        tid = np.asarray(tid_all[i], dtype=np.int64)

        n_hits = x.shape[0]
        if n_hits < args.min_hits:
            skipped_short += 1
            continue

        feat = np.stack([e, x, y, z], axis=-1).astype(np.float32)  # (N,4) = (E,x,y,z)
        reg = np.zeros((n_hits, 7), dtype=np.float32)              # placeholder, see docstring

        features_list.append(feat)
        seg_target_list.append(tid)
        reg_target_list.append(reg)
        kept_hit_counts.append(n_hits)

        if (i + 1) % 500 == 0 or (i + 1) == n_events:
            print(f"[INFO] Processed {i + 1}/{n_events} events "
                  f"(kept {len(features_list)}, skipped-short {skipped_short})")

    n_kept = len(features_list)
    if n_kept == 0:
        sys.exit("[ERROR] No events survived --min-hits filtering. Nothing to write.")

    kept_hit_counts = np.array(kept_hit_counts)
    print(f"[INFO] Kept {n_kept} events. Hits/event: "
          f"min={kept_hit_counts.min()} max={kept_hit_counts.max()} "
          f"mean={kept_hit_counts.mean():.1f} median={np.median(kept_hit_counts):.1f}")

    # shuffle then split train/test
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_kept)
    n_train = int(round(n_kept * args.train_frac))
    train_idx = order[:n_train]
    test_idx = order[n_train:]
    print(f"[INFO] Split: {len(train_idx)} train / {len(test_idx)} test "
          f"(train_frac={args.train_frac})")

    def write_split(name, idx):
        if len(idx) == 0:
            print(f"[WARN] Split '{name}' has 0 events, skipping RaggedMmap creation for it.")
            return
        feat_gen = (features_list[j] for j in idx)
        seg_gen = (seg_target_list[j] for j in idx)
        reg_gen = (reg_target_list[j] for j in idx)

        RaggedMmap.from_generator(
            out_dir=os.path.join(out_dir, f"features_{name}"),
            sample_generator=feat_gen,
            batch_size=min(1000, max(1, len(idx))),
        )
        RaggedMmap.from_generator(
            out_dir=os.path.join(out_dir, f"seg_target_{name}"),
            sample_generator=seg_gen,
            batch_size=min(1000, max(1, len(idx))),
        )
        RaggedMmap.from_generator(
            out_dir=os.path.join(out_dir, f"reg_target_{name}"),
            sample_generator=reg_gen,
            batch_size=min(1000, max(1, len(idx))),
        )
        print(f"[INFO] Wrote '{name}' split: {len(idx)} events -> {out_dir}/{{features,seg_target,reg_target}}_{name}/")

    write_split("train", train_idx)
    write_split("test", test_idx)

    print("[DONE] Conversion complete.")
    print(f"       Output: {out_dir}")
    print(f"       E feature source: {args.e_branch}")
    print(f"       track id source:  {args.track_branch}")


if __name__ == "__main__":
    main()