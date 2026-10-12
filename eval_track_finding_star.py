#!/usr/bin/env python3
"""
STAR-adapted mimic of FM4NPP's real downstream evaluation
(train/downstream/eval_track_finding.py + track_finding_trainer.py's train()/
inference() + trackinghead.py's MambaAttentionHead + loss.py's
PointHungarianMatcher/compute_point_loss/assign_points_to_masks), applied to
STAR data with the frozen sPHENIX-pretrained backbone.

What this replaces vs. evaluate_tracking.py's stand-in:
    evaluate_tracking.py:  frozen embedding -> trained LinearProbe(512,64)
                           -> triplet loss -> KMeans(n_clusters=TRUE n_tracks)
                           -> Hungarian-match clusters to truth.
                           The true track count is handed to the clusterer
                           (oracle-k) -- not how the paper's model works.

    this script:           frozen embedding (ALL 12 backbone layers, not just
                           the last) -> trained MambaAttentionHead (the
                           paper's real DETR/Mask2Former-style set predictor:
                           learned query "prototypes" cross-attend to the
                           hits, each predicts a per-hit mask + a
                           real-track-vs-"no object" class)
                           -> trained with PointHungarianMatcher + Dice/focal/
                           classification losses (loss.py, unmodified)
                           -> decoded with assign_points_to_masks(): each hit
                           goes to whichever query scores highest, or is left
                           unassigned if every query says "no object".
                           The model is never told how many tracks exist.

What's still a simplification vs. the actual paper pipeline:
    - Only the downstream HEAD is trained on STAR; the backbone stays frozen
      (same policy as evaluate_tracking.py / extract_embeddings_linear_probe.py).
      The paper's pipeline is the same in this respect (backbone frozen,
      "downstream adapter" trained) -- see README_workflow.md Step 5.
    - No genuine random-init-BACKBONE ablation exists in the public FM4NPP
      code either (see README_091626.md, "no clean random-init path"
      blocker) -- the "random_init" arm here re-randomizes the BACKBONE
      weights (same as the other probe scripts), then trains a fresh head on
      top, exactly mirroring how the existing pretrained-vs-random-init
      comparison is done elsewhere in this repo.
    - num_prototypes (--max-gt-classes) is a STAR-appropriate value (STAR
      events top out around 4 tracks in the samples looked at so far), not
      the paper's own tuned value -- see configs/star_tracking.yaml.
    - loss/matcher cost weights are FM4NPP's public defaults (also used in
      configs/star_tracking.yaml), not re-tuned for STAR.
    - Regression (momentum/vertex), PID, and noise auxiliary heads that
      MambaAttentionHead defines are left unused (no truth available in the
      STAR conversion for those) -- same reason reg_target/pid_target are
      zero-filled placeholders in convert_star_to_fm4npp.py.

Usage:
    python eval_track_finding_star.py \
        --data-dir data/star_fm4npp_2k \
        --checkpoint checkpoints/pp_nerf_m3_k30.ckpt \
        --n-fit-events 60 --n-train-events 300 --n-test-events 100 \
        --dim-sweep-order 2 1 0 --revert-order 2 1 0 \
        --max-gt-classes 10 --head-epochs 15 \
        --out track_finding_head_summary.png
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

# reuse everything already validated elsewhere in this repo
from extract_embeddings_pca import cartesian_to_polar, compute_star_stats, SPHENIX_STATS
from star_voxelizer import compute_star_final_bins, build_voxelizer, hierarchical_raster_scan_order
from extract_embeddings_linear_probe import event_polar, build_model

# FM4NPP's real downstream-adapter code, used unmodified
FM4NPP_REPO = os.path.expanduser(os.environ.get("FM4NPP_REPO", "~/Work/STAR/PP_collision"))
sys.path.append(FM4NPP_REPO)
sys.path.append(os.path.join(FM4NPP_REPO, "train", "downstream"))
from trackinghead import MambaAttentionHead
from loss import PointHungarianMatcher, compute_point_loss, assign_points_to_masks


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--n-fit-events", type=int, default=60)
    p.add_argument("--n-train-events", type=int, default=300,
                    help="Events used to train the downstream head (MambaAttentionHead).")
    p.add_argument("--n-test-events", type=int, default=100)
    p.add_argument("--min-hits", type=int, default=15)
    p.add_argument("--norm", default="star", choices=["star", "sphenix"])
    p.add_argument("--norm-sample-events", type=int, default=200)
    p.add_argument("--n-bins", type=int, nargs=3, default=[8, 8, 6], metavar=("ETA", "PHI", "RADIUS"))
    p.add_argument("--n-radial-groups", type=int, default=2)
    p.add_argument("--dim-sweep-order", type=int, nargs=3, default=[2, 1, 0], metavar=("D0", "D1", "D2"),
                    help="Default [2,1,0] = FM4NPP's published order='EPR' (see star_voxelizer.py / README_workflow.md).")
    p.add_argument("--revert-order", type=int, nargs=3, default=[2, 1, 0], metavar=("D0", "D1", "D2"))

    # backbone (must match the checkpoint)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--d-state", type=int, default=32)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--klen", type=int, default=30)

    # downstream head (MambaAttentionHead), mirrors track_finding_trainer.py's real constructor call
    p.add_argument("--max-gt-classes", type=int, default=10,
                    help="num_prototypes: upper bound on tracks/event. STAR events sampled so far top out "
                         "around 4 tracks/event (see configs/star_tracking.yaml) -- 10 leaves headroom.")
    p.add_argument("--head-embed-dim", type=int, default=256, help="MambaAttentionHead's internal working dim.")
    p.add_argument("--head-dropout", type=float, default=0.1)
    p.add_argument("--head-epochs", type=int, default=15)
    p.add_argument("--head-lr", type=float, default=1e-4)

    # loss / matcher weights -- FM4NPP public defaults, also used in configs/star_tracking.yaml
    p.add_argument("--loss-matched-ce-weight", type=float, default=0.5)
    p.add_argument("--loss-unmatched-ce-weight", type=float, default=0.1)
    p.add_argument("--loss-dice-weight", type=float, default=1.0)
    p.add_argument("--loss-focal-weight", type=float, default=30.0)

    p.add_argument("--double-majority-threshold", type=float, default=0.5)
    p.add_argument("--decode-threshold", type=float, default=0.0,
                    help="assign_points_to_masks() score threshold below which a hit is left unassigned (-1).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="track_finding_head_summary.png")
    p.add_argument("--feature-cache-dir", default=None,
                    help="Where to disk-cache per-event frozen feature stacks (float16) instead of holding "
                         "them all in RAM across epochs -- large/busy events (e.g. Pythia MinBias, hundreds "
                         "of hits/event) make an in-memory list too big (~40GB+ at a few thousand events). "
                         "Default: a fresh tempdir, cleaned up automatically as each arm/split finishes.")
    p.add_argument("--results-dir", default=None,
                    help="If set, persist everything needed to inspect results or regenerate event displays "
                         "WITHOUT retraining: trained down_model weights (+ backbone weights for the "
                         "random_init arm, since those are randomly generated at construction and otherwise "
                         "unrecoverable), per-test-event results (raw hit coords, true labels, predicted "
                         "assignments, per-event metrics), run config, and summary metrics. One subdirectory "
                         "per arm under this path.")
    p.add_argument("--events-out", default=None,
                    help="If set, save an event-display figure (true-labeled vs. predicted-labeled vs. "
                         "outcome-highlighted hits, plus a true-vs-predicted contingency heatmap) to this path.")
    p.add_argument("--n-display-events", type=int, default=6,
                    help="Number of test events to show in --events-out, spread across the ARI range "
                         "(best/median/worst) rather than cherry-picked.")
    return p.parse_args()


def get_frozen_feature_stack(model, E_n, eta_n, phi_n, r_n, order):
    """Like extract_embeddings_linear_probe.get_frozen_embedding(), but returns ALL backbone
    layer outputs stacked -- (num_layers, N, embed_dim), in ORIGINAL hit order -- matching what
    FM4NPP's real track_finding_trainer.py feeds MambaAttentionHead(pretrain=True):
        _, pre_embed, _ = self.model(grouped, return_z=True)
        feature = torch.stack(pre_embed)
    """
    feat_ordered = np.stack([E_n[order], eta_n[order], phi_n[order], r_n[order]], axis=-1).astype(np.float32)
    x_in = torch.from_numpy(feat_ordered).unsqueeze(0).cuda()
    with torch.no_grad():
        _, feature_layers, _ = model(x_in, return_z=True)  # list of (num_layers) tensors, each (1, N, embed_dim)
    stack_ordered = torch.stack(feature_layers, dim=0).squeeze(1)  # (num_layers, N, embed_dim), ORDERED sequence
    order_t = torch.as_tensor(order, dtype=torch.long, device=stack_ordered.device)
    stack_original = torch.empty_like(stack_ordered)
    stack_original[:, order_t, :] = stack_ordered  # undo reordering, same convention as get_frozen_embedding()
    return stack_original.cpu()  # keep on CPU between events; moved to GPU per-event during train/eval


def save_feature_stack(stack_cpu, path):
    """Disk-cache one event's frozen feature stack as float16 (halves size vs. float32;
    precision loss here is irrelevant -- these are frozen, never-gradient-updated features)."""
    torch.save(stack_cpu.half(), path)


def load_feature_stack(path):
    """Inverse of save_feature_stack(): loads from disk and upcasts back to float32, matching
    what forward_event()/get_frozen_feature_stack() produced originally."""
    return torch.load(path, map_location="cpu").float()


def build_target(track_id, device):
    """Same target format as track_finding_trainer.py's inference()/train() loops."""
    track_id_t = torch.as_tensor(np.asarray(track_id), dtype=torch.long, device=device)
    unique_labels, inverse_indices = torch.unique(track_id_t, sorted=True, return_inverse=True)
    n_gt = unique_labels.numel()
    one_hot = torch.nn.functional.one_hot(inverse_indices, num_classes=n_gt).float()  # (N, n_gt)
    target = {
        "masks": one_hot.permute(1, 0),                       # (n_gt, N)
        "labels": torch.ones(n_gt, dtype=torch.long, device=device),  # class 1 = "real track"
    }
    return target, n_gt


def forward_event(down_model, feature_stack_cpu, device):
    """feature_stack_cpu: (num_layers, N, embed_dim) on CPU, ORIGINAL hit order."""
    feature = feature_stack_cpu.to(device).unsqueeze(1)  # (num_layers, 1, N, embed_dim)
    n_hits = feature.shape[2]
    dummy_x = torch.zeros(1, n_hits, 4, device=device)  # unused when pretrain=True (see trackinghead.py forward())
    padding_mask = torch.ones(1, n_hits, dtype=torch.bool, device=device)  # no padding: single event, all real hits
    pred_dict = down_model(dummy_x, feature, padding_mask=padding_mask, pretrain=True)
    outputs = {
        "pred_probs": pred_dict["class_probs"],                    # (1, num_prototypes, 2)
        "pred_masks": pred_dict["mask_probs"].permute(0, 2, 1),    # (1, num_prototypes, N)
    }
    return outputs, padding_mask


def train_head(down_model, matcher, feature_paths, labels, epochs, lr, device, weights):
    w_ce, w_unce, w_dice, w_focal = weights
    opt = torch.optim.AdamW(down_model.parameters(), lr=lr, weight_decay=1e-4)
    down_model.train()
    for ep in range(epochs):
        total, n_seen = 0.0, 0
        order = np.random.permutation(len(feature_paths))
        for idx in order:
            feat_stack, lab = load_feature_stack(feature_paths[idx]), labels[idx]
            target, n_gt = build_target(lab, device)
            if n_gt < 1:
                continue
            outputs, padding_mask = forward_event(down_model, feat_stack, device)
            losses = compute_point_loss(outputs, [target], padding_mask, matcher, no_object_class=0)
            loss = (losses["loss_matched_ce"] * w_ce + losses["loss_unmatched_ce"] * w_unce
                    + losses["loss_dice"] * w_dice + losses["loss_focal"] * w_focal)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n_seen += 1
        if n_seen and (ep % 2 == 0 or ep == epochs - 1):
            print(f"    [head] epoch {ep}: mean weighted loss = {total / n_seen:.4f}")
    down_model.eval()
    return down_model


def score_event(down_model, feat_stack, lab, device, decode_threshold, dm_threshold):
    """Decode via the real, non-oracle assign_points_to_masks() and score with ARI + the
    same double-majority efficiency/purity convention evaluate_tracking.py uses, so numbers
    are comparable across the two evaluation methodologies."""
    from sklearn.metrics import adjusted_rand_score
    with torch.no_grad():
        outputs, _ = forward_event(down_model, feat_stack, device)
        decoded = assign_points_to_masks(outputs, no_object_class=0, option=2, threshold=decode_threshold)
    assignments = decoded["assignments"][0].cpu().numpy()  # (N,), -1 == "no track" (every query said no-object)
    true_labels = np.asarray(lab)
    ari = float(adjusted_rand_score(true_labels, assignments))

    true_ids = np.unique(true_labels)
    n_tracks = len(true_ids)
    pred_ids = np.unique(assignments[assignments >= 0])
    n_found = 0
    correct_mask = np.zeros(len(true_labels), dtype=bool)
    for pid_ in pred_ids:
        pred_mask = assignments == pid_
        vals, counts = np.unique(true_labels[pred_mask], return_counts=True)
        matched_true_id = vals[np.argmax(counts)]
        true_mask = true_labels == matched_true_id
        intersection = (pred_mask & true_mask).sum()
        efficiency = intersection / true_mask.sum()
        purity = intersection / pred_mask.sum()
        if efficiency >= dm_threshold and purity >= dm_threshold:
            n_found += 1
            correct_mask |= pred_mask & true_mask
    frac_found = n_found / n_tracks if n_tracks else float("nan")
    n_unassigned = int((assignments < 0).sum())
    return dict(ari=ari, frac_found=frac_found, hit_frac_found=float(correct_mask.mean()),
                n_unassigned=n_unassigned, n_hits=len(true_labels),
                assignments=assignments, correct_mask=correct_mask)


def save_arm_results(results_dir, arm_label, down_model, backbone_model, is_random_init, test_records):
    """Persist everything needed to inspect this arm's results, or regenerate event displays,
    without rerunning training or the backbone. See --results-dir help text for what/why."""
    arm_dir = os.path.join(results_dir, arm_label)
    os.makedirs(arm_dir, exist_ok=True)
    torch.save(down_model.state_dict(), os.path.join(arm_dir, "down_model.pt"))
    if is_random_init:
        # the pretrained arm's backbone is just the known checkpoint path (already on disk, recorded in
        # config.json) -- but random_init's backbone weights were generated fresh at construction and are
        # otherwise unrecoverable, so they must be saved explicitly to make this arm reproducibly reloadable.
        torch.save(backbone_model.state_dict(), os.path.join(arm_dir, "backbone.pt"))
    with open(os.path.join(arm_dir, "test_results.pkl"), "wb") as f:
        pickle.dump(test_records, f)
    print(f"[INFO] Saved {arm_label} results to {arm_dir}/ "
          f"(down_model.pt{' + backbone.pt' if is_random_init else ''} + test_results.pkl, "
          f"{len(test_records)} test-event records)")


def select_display_events(test_records, n_events, forced_event_ids=None):
    """Pick a spread across the ARI range (worst/median/best), not cherry-picked successes.

    If forced_event_ids is given, use exactly those events (in that order) instead of re-ranking
    by this call's own ARI values -- needed so the SAME events show up in both arms' figures.
    Picking independently per arm (by each arm's own ARI ranking) means pretrained and random_init
    almost never land on the same events at the same "worst/median/best" position, since their ARI
    values differ per event -- defeats the point of a side-by-side comparison."""
    if forced_event_ids is not None:
        by_id = {r["event_idx"]: r for r in test_records}
        return [by_id[eid] for eid in forced_event_ids if eid in by_id]
    order = np.argsort([r["ari"] for r in test_records])
    n = len(order)
    if n <= n_events:
        return [test_records[i] for i in order]
    # evenly spaced indices through the sorted (worst->best) order, including both extremes
    pick_pos = np.linspace(0, n - 1, n_events).round().astype(int)
    pick_pos = sorted(set(pick_pos.tolist()))
    return [test_records[order[p]] for p in pick_pos]


def plot_event_displays(test_records, out_path, n_events, arm_label, forced_event_ids=None):
    """Three panels per selected event: true-track-colored, predicted-cluster-colored, and
    outcome-highlighted (correct / wrongly-assigned / unassigned) hits in the x-y plane."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = select_display_events(test_records, n_events, forced_event_ids=forced_event_ids)
    cmap = plt.get_cmap("tab20")

    fig, axes = plt.subplots(len(picks), 3, figsize=(13, 3.6 * len(picks)), squeeze=False)
    for row, rec in enumerate(picks):
        feat, true_labels, assignments = rec["feat"], rec["true_labels"], rec["assignments"]
        cx, cy = feat[:, 1], feat[:, 2]
        n_tracks = len(np.unique(true_labels))

        ax = axes[row, 0]
        for k, tid in enumerate(np.unique(true_labels)):
            m = true_labels == tid
            ax.scatter(cx[m], cy[m], s=12, color=cmap(k % 20))
        ax.set_title(f"event {rec['event_idx']}: TRUE ({n_tracks} tracks)")

        ax = axes[row, 1]
        pred_ids = np.unique(assignments)
        for k, pid_ in enumerate(pred_ids):
            m = assignments == pid_
            if pid_ < 0:
                ax.scatter(cx[m], cy[m], s=12, color="lightgray", marker="x", label="unassigned")
            else:
                ax.scatter(cx[m], cy[m], s=12, color=cmap(k % 20))
        ax.set_title(f"PREDICTED ({len(pred_ids[pred_ids >= 0])} found, ARI={rec['ari']:.3f})")

        ax = axes[row, 2]
        correct_mask = rec["correct_mask"]
        wrong_mask = (assignments >= 0) & ~correct_mask
        unassigned_mask = assignments < 0
        ax.scatter(cx, cy, s=10, color="steelblue", label="all hits", zorder=1)
        ax.scatter(cx[wrong_mask], cy[wrong_mask], s=28, color="orange", marker="x", label="wrong", zorder=2)
        ax.scatter(cx[unassigned_mask], cy[unassigned_mask], s=28, color="gray", marker=".", label="unassigned", zorder=2)
        ax.scatter(cx[correct_mask], cy[correct_mask], s=28, color="green", marker="x", label="correct", zorder=3)
        ax.set_title(f"OUTCOME (frac_found={rec['frac_found']:.3f})")
        if row == 0:
            ax.legend(fontsize=7, loc="upper right")

        for a in axes[row]:
            a.set_xlabel("x [cm]"); a.set_ylabel("y [cm]"); a.set_aspect("equal", adjustable="datalim")

    fig.suptitle(f"{arm_label}: event displays, spread across ARI range (not cherry-picked)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[INFO] Saved event-display figure to {out_path}")
    return [rec["event_idx"] for rec in picks]  # so the caller can reuse the same picks for other arms


def plot_contingency_heatmaps(test_records, out_path, arm_label, n_events=4,
                               min_true_tracks=8, max_true_tracks=20):
    """True-track x predicted-cluster hit-count heatmaps -- picks events in a MODERATE multiplicity
    band (default 8-20 true tracks), not the single busiest ones. Tried busiest-first (up to ~55
    simultaneous tracks) and it was illegible regardless of styling -- that many thin rows/columns
    just can't render as a readable grid, annotated or not; this isn't fixable with better colors,
    it's the wrong event range for this representation. A low-track event (<8) collapses to 1-2
    cells and is better read directly off the scatter plot instead."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_true_tracks = [len(np.unique(r["true_labels"])) for r in test_records]
    candidates = [i for i, n in enumerate(n_true_tracks) if min_true_tracks <= n <= max_true_tracks]
    if len(candidates) < n_events:
        candidates = [i for i, n in enumerate(n_true_tracks) if n >= min_true_tracks]  # widen if too few
    if not candidates:
        print(f"[INFO] No events in the {min_true_tracks}-{max_true_tracks}-track band for "
              f"contingency heatmap ({out_path}), skipping.")
        return
    order = sorted(candidates, key=lambda i: n_true_tracks[i])[:n_events]  # smallest-in-band first
    picks = [test_records[i] for i in order]
    if not picks:
        print(f"[INFO] No events available for contingency heatmap ({out_path}), skipping.")
        return

    fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 5.2), squeeze=False)
    for col, rec in enumerate(picks):
        true_labels, assignments = rec["true_labels"], rec["assignments"]
        true_ids = np.unique(true_labels)
        # drop "unassigned" (-1) as its own column -- it's not a track, would just add visual noise;
        # those hits already show up clearly as gray in the scatter plot's outcome panel.
        pred_ids = np.unique(assignments[assignments >= 0])
        cm = np.zeros((len(true_ids), len(pred_ids)), dtype=int)
        for ti, t in enumerate(true_ids):
            for pi, p in enumerate(pred_ids):
                cm[ti, pi] = ((true_labels == t) & (assignments == p)).sum()
        ax = axes[0, col]
        im = ax.imshow(cm, aspect="auto", cmap="Blues")
        annotate = cm.shape[0] * cm.shape[1] <= 400  # keep text legible; skip for very large grids
        if annotate:
            vmax = cm.max() if cm.size else 1
            for ti in range(cm.shape[0]):
                for pi in range(cm.shape[1]):
                    v = cm[ti, pi]
                    if v > 0:
                        ax.text(pi, ti, str(v), ha="center", va="center", fontsize=7,
                                 color="white" if v > vmax * 0.6 else "black")
        ax.set_xticks(range(len(pred_ids))); ax.set_xticklabels(range(len(pred_ids)), fontsize=7)
        ax.set_yticks(range(len(true_ids))); ax.set_yticklabels(range(len(true_ids)), fontsize=7)
        ax.set_xlabel(f"predicted cluster (of {len(pred_ids)} found)")
        ax.set_ylabel(f"true track (of {len(true_ids)})")
        ax.set_title(f"event {rec['event_idx']}: ARI={rec['ari']:.3f}\n"
                      f"{'annotated hit counts' if annotate else 'too busy to annotate -- color only'}")
    fig.suptitle(f"{arm_label}: true x predicted hit-count contingency, moderate-multiplicity events "
                 f"({min_true_tracks}-{max_true_tracks} true tracks) -- a clean row/column = a well-separated track")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[INFO] Saved contingency heatmap figure to {out_path}")


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
    features_mmap = RaggedMmap(feat_path)
    target_mmap = RaggedMmap(tgt_path)
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

    cache_root = os.path.expanduser(args.feature_cache_dir) if args.feature_cache_dir else tempfile.mkdtemp(prefix="fm4npp_feat_cache_")
    os.makedirs(cache_root, exist_ok=True)
    print(f"[INFO] Disk-caching frozen features under {cache_root} (avoids holding them all in RAM).")

    def collect_features(idx_list, model, cache_subdir):
        cache_dir = os.path.join(cache_root, cache_subdir)
        os.makedirs(cache_dir, exist_ok=True)
        paths, labels = [], []
        for i in idx_list:
            feat = np.asarray(features_mmap[i])
            track_id = np.asarray(target_mmap[i])
            E_n, eta_n, phi_n, r_n, r_raw = event_polar(feat, stats)
            order = hierarchical_raster_scan_order(voxelizer, eta_n, phi_n, r_n)
            stack = get_frozen_feature_stack(model, E_n, eta_n, phi_n, r_n, order)
            path = os.path.join(cache_dir, f"{i}.pt")
            save_feature_stack(stack, path)
            paths.append(path)
            labels.append(track_id)
        return paths, labels

    weights = (args.loss_matched_ce_weight, args.loss_unmatched_ce_weight,
               args.loss_dice_weight, args.loss_focal_weight)

    results = {}
    shared_display_event_ids = None  # set after the first arm, reused so both arms show the SAME events
    for arm_label, load_ckpt in [("pretrained", True), ("random_init", False)]:
        print(f"\n[INFO] === {arm_label} ===")
        model = build_model(args.embed_dim, args.num_layers, args.d_state, args.d_conv, args.expand,
                            args.klen, args.checkpoint, load_ckpt)

        print(f"[INFO] Collecting frozen features ({args.num_layers} layers) for {len(train_idx)} train events...")
        train_paths, train_labels = collect_features(train_idx, model, f"{arm_label}_train")

        down_model = MambaAttentionHead(
            input_dim=args.embed_dim, embed_dim=args.head_embed_dim, num_layers=0,
            num_feature_layers=args.num_layers, num_prototypes=args.max_gt_classes,
            dropout=args.head_dropout,
        ).to(device)
        n_params = sum(p.numel() for p in down_model.parameters())
        print(f"[INFO] MambaAttentionHead built: num_prototypes={args.max_gt_classes}, "
              f"head_embed_dim={args.head_embed_dim}, {n_params:,} trainable params.")

        matcher = PointHungarianMatcher(cost_class=args.loss_matched_ce_weight,
                                         cost_dice=args.loss_dice_weight,
                                         cost_focal=args.loss_focal_weight)

        print(f"[INFO] Training downstream head for {args.head_epochs} epochs "
              f"(backbone frozen throughout)...")
        down_model = train_head(down_model, matcher, train_paths, train_labels,
                                 args.head_epochs, args.head_lr, device, weights)
        shutil.rmtree(os.path.join(cache_root, f"{arm_label}_train"), ignore_errors=True)  # done with it, free disk

        print(f"[INFO] Collecting frozen features for {len(test_idx)} test events...")
        test_paths, test_labels = collect_features(test_idx, model, f"{arm_label}_test")

        aris, fracs, hit_fracs, unassigned_fracs = [], [], [], []
        test_records = []
        for event_idx, feat_path, lab in zip(test_idx, test_paths, test_labels):
            feat_stack = load_feature_stack(feat_path)
            r = score_event(down_model, feat_stack, lab, device,
                            args.decode_threshold, args.double_majority_threshold)
            aris.append(r["ari"]); fracs.append(r["frac_found"]); hit_fracs.append(r["hit_frac_found"])
            unassigned_fracs.append(r["n_unassigned"] / r["n_hits"])
            if args.results_dir or args.events_out:
                # raw (E,x,y,z) straight from the mmap -- needed for event displays, never saved during
                # feature collection since that path only persists backbone features, not coordinates.
                raw_feat = np.asarray(features_mmap[event_idx])
                test_records.append(dict(
                    event_idx=int(event_idx), feat=raw_feat, true_labels=np.asarray(lab),
                    assignments=r["assignments"], correct_mask=r["correct_mask"],
                    ari=r["ari"], frac_found=r["frac_found"], hit_frac_found=r["hit_frac_found"],
                ))
        shutil.rmtree(os.path.join(cache_root, f"{arm_label}_test"), ignore_errors=True)

        results[arm_label] = dict(ari=np.array(aris), frac_found=np.array(fracs),
                                   hit_frac=np.array(hit_fracs), unassigned=np.array(unassigned_fracs))
        print(f"[RESULT] {arm_label}: ARI = {np.mean(aris):.4f} +/- {np.std(aris):.4f}, "
              f"frac_found (double-majority) = {np.mean(fracs):.4f} +/- {np.std(fracs):.4f}, "
              f"mean unassigned-hit fraction = {np.mean(unassigned_fracs):.4f}  (n={len(aris)} events)")

        if args.results_dir:
            save_arm_results(os.path.expanduser(args.results_dir), arm_label, down_model, model,
                              is_random_init=(arm_label == "random_init"), test_records=test_records)
        if args.events_out:
            base, ext = os.path.splitext(os.path.expanduser(args.events_out))
            chosen_ids = plot_event_displays(test_records, f"{base}_{arm_label}{ext}", args.n_display_events,
                                              arm_label, forced_event_ids=shared_display_event_ids)
            if shared_display_event_ids is None:
                shared_display_event_ids = chosen_ids  # lock in pretrained's picks for random_init to reuse
            plot_contingency_heatmaps(test_records, f"{base}_{arm_label}_contingency{ext}", arm_label)

    print("\n=== SUMMARY (real query-based decode, no oracle track count) ===")
    summary = {}
    for metric in ("ari", "frac_found"):
        p_arr, r_arr = results["pretrained"][metric], results["random_init"][metric]
        n = min(len(p_arr), len(r_arr))
        p_arr, r_arr = p_arr[:n], r_arr[:n]
        deltas = p_arr - r_arr
        print(f"\n-- {metric} --")
        print(f"pretrained : {p_arr.mean():.4f} (std={p_arr.std():.4f})")
        print(f"random_init: {r_arr.mean():.4f} (std={r_arr.std():.4f})")
        print(f"delta      : {deltas.mean():+.4f} (std={deltas.std():.4f}, n={n})")
        metric_summary = dict(pretrained_mean=float(p_arr.mean()), pretrained_std=float(p_arr.std()),
                               random_init_mean=float(r_arr.mean()), random_init_std=float(r_arr.std()),
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
        print(f"[INFO] Saved summary.json to {os.path.expanduser(args.results_dir)}/")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in zip(axes, ("ari", "frac_found"), ("ARI", "double-majority frac_found")):
        p_arr, r_arr = results["pretrained"][metric], results["random_init"][metric]
        ax.boxplot([p_arr, r_arr], tick_labels=["pretrained", "random-init"])
        ax.set_title(f"{title}\n(n={min(len(p_arr), len(r_arr))} test events)")
        ax.set_ylabel(title)
    fig.suptitle("Real query-based head (MambaAttentionHead), oracle-free decode")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\n[DONE] Saved summary figure to {args.out}")
    if not args.feature_cache_dir:
        shutil.rmtree(cache_root, ignore_errors=True)  # only auto-clean tempdirs we created ourselves


if __name__ == "__main__":
    main()
