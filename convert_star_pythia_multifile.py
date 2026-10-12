#!/usr/bin/env python3
"""
Multi-file STAR TPC Pythia MinBias -> FM4NPP RaggedMmap converter, with REAL
per-hit particle-species labels (pid_target) -- unlike convert_star_to_fm4npp.py,
which always writes pid_target as zeros because the single-big-file dataset it
was built for never had gePid/mcP/mcPt/mcEta/mcVr populated (verified: all
-1/-999 placeholders in that production).

This new dataset (/home/daq/Work/STAR/files/pythia8_pp510_minbias/) is split
across many small files (0000.tpchits.root ... NNNN.tpchits.root), each with
two trees:
    T  : per-event TPC hits (same schema as before: x,y,z,q,adc,...,IdTruth)
    TT : per-track truth info, keyed by an integer `evt` field -- confirmed
         (by direct inspection, not assumed) to align with T's entries as
         T[i] (0-based) <-> TT rows where evt == i+1 (1-based), with full
         hit-track-id overlap in every file checked. TT also has REAL
         (non-placeholder) gePid (species), mcP/mcPt/mcEta (truth momentum),
         mcVr (truth vertex radius) in this production.

Per hit, this script:
  1. Looks up its track's gePid via the verified T<->TT join (idTruth).
  2. Recodes the raw GEANT3 code into a compact 6-class scheme (see
     GEANT_TO_CLASS below) -- collapsing the long tail (antiprotons, light
     nuclei, anything unmatched) into an "other" class, since each of those
     is <1% of hit-producing tracks in this sample.
  3. (Bonus, since the data is right there) also writes REAL reg_target =
     (mcP, mcPt, mcEta, mcVr, 0, 0, 0) per hit -- zero-padded to the same
     7-wide layout convert_star_to_fm4npp.py already uses, so TPCBatchDataset
     can load it unmodified. Not used by anything in this repo yet; this is
     groundwork for a possible future track-fitting experiment, not required
     for the PID task this script was built for.

Verified during development (not assumed):
  - T[i] <-> TT evt=i+1 alignment, checked per-file, not just once.
  - Duplicate idTruth rows within one event's TT rows exist, but always
    agree on gePid where checked (331/331 duplicate groups inspected) --
    so a simple last-write-wins dict join is safe for species lookup.

Usage:
    python convert_star_pythia_multifile.py \
        --input-dir /home/daq/Work/STAR/files/pythia8_pp510_minbias \
        --output-dir data/pythia510_multifile_8k \
        --max-events 8000 --train-frac 0.8 --min-hits 15
"""

import argparse
import glob
import os
import shutil
import sys

import numpy as np

# GEANT3 particle codes observed in this production's hit-producing tracks:
# {2,3,5,6,8,9,11,12,14,15,45,46,47} = {e+,e-,mu+,mu-,pi+,pi-,K+,K-,p,pbar,d,t,alpha}
# Collapsed into 6 compact classes -- the long tail (pbar, d, t, alpha; each
# <1% of hit-producing TRACKS, see plan discussion) goes to "other".
#
# Separately, verified (not assumed): ~14% of HITS belong to track ids that
# never appear in TT's truth table for their event at all -- not rare species,
# just no truth-matched track recorded (TT only lists reconstructed-quality
# tracks; T has every raw hit regardless). Conflating "genuinely rare
# species" with "no truth available" into one "other" bucket would be
# misleading -- these get their own class 6 "unknown" and should be EXCLUDED
# from PID training/eval (no real label exists), not treated as a prediction
# target.
PID_CLASS_NAMES = ["pion", "muon", "electron", "proton", "kaon", "other", "unknown"]
GEANT_TO_CLASS = {
    8: 0, 9: 0,     # pi+, pi-
    5: 1, 6: 1,     # mu+, mu-
    2: 2, 3: 2,     # e+, e-
    14: 3,          # proton
    11: 4, 12: 4,   # K+, K-
    # everything else matched-but-rare (15 pbar, 45/46/47 d/t/alpha) -> class 5 "other"
}
OTHER_CLASS = 5
UNKNOWN_CLASS = 6  # no T<->TT truth match for this hit's track id -- exclude from PID supervision


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, help="Directory of *.tpchits.root files")
    p.add_argument("--file-pattern", default="*.tpchits.root")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-events", type=int, default=8000,
                    help="Stop once this many events (post min-hits filter) are collected. "
                         "Default 8000 -- same order of magnitude as prior experiments in this repo.")
    p.add_argument("--max-files", type=int, default=None,
                    help="Optional cap on number of input files scanned (for quick tests).")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-hits", type=int, default=15)
    p.add_argument("--x-branch", default="TPCHits_x")
    p.add_argument("--y-branch", default="TPCHits_y")
    p.add_argument("--z-branch", default="TPCHits_z")
    p.add_argument("--e-branch", default="TPCHits_adc")
    p.add_argument("--track-branch", default="TPCHits_IdTruth")
    p.add_argument("--overwrite", action="store_true")
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
            sys.exit(f"[ERROR] {out_dir} already exists. Use --overwrite or pick a different --output-dir.")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.input_dir), args.file_pattern)))
    if args.max_files is not None:
        files = files[:args.max_files]
    if not files:
        sys.exit(f"[ERROR] no files matching {args.file_pattern} in {args.input_dir}")
    print(f"[INFO] {len(files)} input files found.")

    features_list, seg_target_list, reg_target_list, pid_target_list = [], [], [], []
    kept_hit_counts = []
    skipped_short = 0
    n_files_used = 0
    n_events_no_tt_match = 0

    for fi, fpath in enumerate(files):
        if len(features_list) >= args.max_events:
            break
        n_files_used += 1
        f = uproot.open(fpath)
        T = f["T"]
        TT = f["TT"]

        tt_arrays = TT.arrays(["evt", "idTruth", "gePid", "mcP", "mcPt", "mcEta", "mcVr"], library="np")
        # build per-event {idTruth: (gePid, mcP, mcPt, mcEta, mcVr)} truth maps
        evt_maps = {}
        for j in range(len(tt_arrays["evt"])):
            e = int(tt_arrays["evt"][j])
            d = evt_maps.setdefault(e, {})
            d[int(tt_arrays["idTruth"][j])] = (
                int(tt_arrays["gePid"][j]), float(tt_arrays["mcP"][j]), float(tt_arrays["mcPt"][j]),
                float(tt_arrays["mcEta"][j]), float(tt_arrays["mcVr"][j]),
            )

        branches = [args.x_branch, args.y_branch, args.z_branch, args.e_branch, args.track_branch]
        t_arrays = T.arrays(branches, library="np")
        n_events = T.num_entries

        for i in range(n_events):
            if len(features_list) >= args.max_events:
                break
            x = np.asarray(t_arrays[args.x_branch][i], dtype=np.float32)
            y = np.asarray(t_arrays[args.y_branch][i], dtype=np.float32)
            z = np.asarray(t_arrays[args.z_branch][i], dtype=np.float32)
            e_feat = np.asarray(t_arrays[args.e_branch][i], dtype=np.float32)
            tid = np.asarray(t_arrays[args.track_branch][i], dtype=np.int64)

            n_hits = x.shape[0]
            if n_hits < args.min_hits:
                skipped_short += 1
                continue

            evt_id = i + 1  # verified T[i] <-> TT evt=i+1 convention
            truth_map = evt_maps.get(evt_id)
            if truth_map is None:
                n_events_no_tt_match += 1

            pid = np.full((n_hits,), UNKNOWN_CLASS, dtype=np.int64)
            reg = np.zeros((n_hits, 7), dtype=np.float32)
            if truth_map is not None:
                for k in range(n_hits):
                    entry = truth_map.get(int(tid[k]))
                    if entry is None:
                        continue  # no truth-matched track for this hit -- stays UNKNOWN_CLASS
                    gepid, mcp, mcpt, mceta, mcvr = entry
                    pid[k] = GEANT_TO_CLASS.get(gepid, OTHER_CLASS)
                    reg[k, 0] = mcp
                    reg[k, 1] = mcpt
                    reg[k, 2] = mceta
                    reg[k, 3] = mcvr

            feat = np.stack([e_feat, x, y, z], axis=-1).astype(np.float32)  # (N,4) = (E,x,y,z)
            features_list.append(feat)
            seg_target_list.append(tid)
            reg_target_list.append(reg)
            pid_target_list.append(pid)
            kept_hit_counts.append(n_hits)

        if (fi + 1) % 50 == 0 or len(features_list) >= args.max_events:
            print(f"[INFO] Scanned {fi + 1}/{len(files)} files, kept {len(features_list)} events "
                  f"(skipped-short {skipped_short}, no-TT-match {n_events_no_tt_match})")

    n_kept = len(features_list)
    if n_kept == 0:
        sys.exit("[ERROR] No events survived filtering. Nothing to write.")

    kept_hit_counts = np.array(kept_hit_counts)
    print(f"[INFO] Used {n_files_used} files. Kept {n_kept} events. Hits/event: "
          f"min={kept_hit_counts.min()} max={kept_hit_counts.max()} "
          f"mean={kept_hit_counts.mean():.1f} median={np.median(kept_hit_counts):.1f}")
    print(f"[INFO] Events with no TT truth match at all: {n_events_no_tt_match} "
          f"(their hits all fall back to pid_target=UNKNOWN_CLASS={UNKNOWN_CLASS}, reg_target=0)")

    all_pid = np.concatenate(pid_target_list)
    print("[INFO] pid_target class balance (per-hit):")
    for c, name in enumerate(PID_CLASS_NAMES):
        frac = (all_pid == c).mean()
        tag = "  <- exclude from PID training/eval, no real label" if c == UNKNOWN_CLASS else ""
        print(f"       {c} {name:10s}: {frac*100:5.1f}%{tag}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_kept)
    n_train = int(round(n_kept * args.train_frac))
    train_idx = order[:n_train]
    test_idx = order[n_train:]
    print(f"[INFO] Split: {len(train_idx)} train / {len(test_idx)} test (train_frac={args.train_frac})")

    def write_split(name, idx):
        if len(idx) == 0:
            print(f"[WARN] Split '{name}' has 0 events, skipping.")
            return
        RaggedMmap.from_generator(
            out_dir=os.path.join(out_dir, f"features_{name}"),
            sample_generator=(features_list[j] for j in idx),
            batch_size=min(1000, max(1, len(idx))),
        )
        RaggedMmap.from_generator(
            out_dir=os.path.join(out_dir, f"seg_target_{name}"),
            sample_generator=(seg_target_list[j] for j in idx),
            batch_size=min(1000, max(1, len(idx))),
        )
        RaggedMmap.from_generator(
            out_dir=os.path.join(out_dir, f"reg_target_{name}"),
            sample_generator=(reg_target_list[j] for j in idx),
            batch_size=min(1000, max(1, len(idx))),
        )
        RaggedMmap.from_generator(
            out_dir=os.path.join(out_dir, f"pid_target_{name}"),
            sample_generator=(pid_target_list[j] for j in idx),
            batch_size=min(1000, max(1, len(idx))),
        )
        print(f"[INFO] Wrote '{name}' split: {len(idx)} events -> {out_dir}/{{features,seg_target,reg_target,pid_target}}_{name}/")

    write_split("train", train_idx)
    write_split("test", test_idx)

    print("[DONE] Conversion complete.")
    print(f"       Output: {out_dir}")
    print(f"       pid_target classes: {list(enumerate(PID_CLASS_NAMES))}")
    print(f"       reg_target columns: [mcP, mcPt, mcEta, mcVr, 0, 0, 0]")


if __name__ == "__main__":
    main()
