# STAR × FM4NPP: End-to-End Workflow

A step-by-step walkthrough of this repo's pipeline, written for those who
want to understand *what happens to a hit* between "we have a ROOT file of
simulated STAR TPC hits" and "here's a plot of which hits got reconstructed
into which track." Each step below explains: what goes in, what comes out,
what it's *for*, and a short snippet to reproduce/inspect that output
yourself — you shouldn't have to take any of this on faith.

For the full repo reference (environment setup, all CLI flags, results) see
[README.md](README.md). For the physics *result* this pipeline produced, see
[README_082026.md](README_082026.md) (embedding-quality transfer) and
[README_091626.md](README_091626.md) (tracking-level evaluation).

All snippets assume you've done the one-time setup from README.md's
[Environment setup](README.md#environment-setup):

```bash
source ~/fm4npp-env/bin/activate
export PYTHONPATH=/path/to/PP_collision:$PYTHONPATH   # FM4NPP's own repo
cd /home/daq/Work/STAR/fm4npp4starTPC                  # this repo
```

## The pipeline at a glance

```
merged_TPCHitsTree.root                           (raw STAR simulation, one entry per event)
        │  convert_star_to_fm4npp.py
        ▼
features_{train,test}/, seg_target_{train,test}/  (RaggedMmap: per-hit (E,x,y,z) + true track id)
        │  star_voxelizer.py  (fit bins on a handful of events)
        ▼
a fitted Voxelizer  (in memory: tells you the hit ORDER to feed each event into Mamba in)
        │  extract_embeddings_linear_probe.py
        ▼
per-hit embeddings → trained linear probe → silhouette score  (is the embedding separable by track?)
        │  evaluate_tracking.py
        ▼
per-event clustering + Hungarian match → ARI, efficiency, purity, "found" tracks
        ▼
hit distribution plots: generated hits (blue) vs. reconstructed hits (red crosses)
```

Each stage is its own script/module in this repo, runnable independently once
its inputs exist — you ~~don't~~ have to run the whole chain to inspect one link
of it.

---

## Step 0: Input — the raw STAR simulation

**What it is:** a ROOT `TTree` (`merged_TPCHitsTree.root`, tree name `T`), one
entry per collision event, with jagged (variable-length) branches — one array
entry per hit in that event:

| branch | meaning |
|---|---|
| `TPCHits_x`, `_y`, `_z` | hit position, cm, STAR TPC frame |
| `TPCHits_adc` | raw ADC ionization signal (this becomes FM4NPP's "E" feature) |
| `TPCHits_q` | calibrated charge (alternative to `_adc`, not used by default) |
| `TPCHits_IdTruth` | Monte Carlo truth track ID for that hit — this is "ground truth" for everything downstream |
| `TPCHits_pad`, `_row`, `_sector`, `_timebucket` | raw detector readout coordinates, not used by this pipeline |

**Touch it yourself** — open the file, look at one event's raw hits, no
FM4NPP code involved at all (**only file with 100 events available here**, adjust correspondingly or generate some):

```python
import uproot
import matplotlib.pyplot as plt

tree = uproot.open("tests/fixtures/test_100events.root")["T"]
arrs = tree.arrays(["TPCHits_x", "TPCHits_y", "TPCHits_IdTruth"], entry_stop=1, library="np")
x, y, tid = arrs["TPCHits_x"][0], arrs["TPCHits_y"][0], arrs["TPCHits_IdTruth"][0]

plt.scatter(x, y, c=tid, cmap="tab10", s=10)
plt.xlabel("x (cm)"); plt.ylabel("y (cm)")
plt.title(f"event 0: {len(x)} hits, {len(set(tid))} true tracks")
plt.savefig("event0_raw.png")
```

That's the ground truth every later metric (silhouette, ARI, efficiency,
purity) is measured against.

---

## Step 1: Convert to FM4NPP's format — `convert_star_to_fm4npp.py`

**What it does:** FM4NPP's model expects a specific per-hit feature layout
and a specific on-disk format (`RaggedMmap` — a memory-mapped, variable-length
array store from the `mmap_ninja` package, so you can randomly access event
`i`'s hits without loading the whole dataset into RAM). This script builds
that from the raw ROOT branches:

- **features** `(N_hits, 4)` = `(E, x, y, z)` — `E` comes from `TPCHits_adc`
  by default, matching the ADC-signal convention FM4NPP's pretrained
  normalization constants were tuned against on sPHENIX.
- **seg_target** `(N_hits,)` = `TPCHits_IdTruth` — the true track ID per hit,
  used later as the label for silhouette/ARI/efficiency, never fed to the
  model itself.
- **reg_target** `(N_hits, 7)` = all zeros — a placeholder; FM4NPP's dataset
  loader expects this array to exist, but nothing in this repo trains with it.

Events are shuffled and split into `train`/`test` subsets (80/20 by default).

```bash
python convert_star_to_fm4npp.py \
    --input data/merged_TPCHitsTree.root \
    --tree T \
    --output-dir data/star_fm4npp_2k \
    --max-events 2000 --train-frac 0.8 --overwrite
```

**Output:** `data/star_fm4npp_2k/{features,seg_target,reg_target}_{train,test}/`
(RaggedMmap directories).

**Touch it yourself:**

```python
from mmap_ninja import RaggedMmap
import numpy as np

feat = RaggedMmap("data/star_fm4npp_2k/features_train")
tgt  = RaggedMmap("data/star_fm4npp_2k/seg_target_train")

f = np.asarray(feat[0])   # (N_hits, 4) = (E, x, y, z)
t = np.asarray(tgt[0])    # (N_hits,)   = true track id per hit
print(f.shape, "hits;", len(np.unique(t)), "true tracks")
```

`test_convertion.py` in this repo is meant to be exactly this sanity check —
**known issue:** it currently hardcodes a stale absolute path from before this
repo was reorganized (`/home/daq/Work/STAR/data/star_fm4npp_2k`, which no
longer exists) instead of the relative `data/star_fm4npp_2k` used everywhere
else. Use the snippet above until it's fixed.

**Reproducibility note, found the hard way:** `--max-events` controls how
much of the source ROOT file gets converted, which in turn controls how many
`train`/`test` events later steps have to draw from — it is *not* just a
speed knob. This repo's headline numbers (README_082026.md, README_091626.md)
were computed against a conversion with **~3118 train events**; the
`--max-events 2000` example above (and in the [Quick reference](#quick-reference-running-the-full-chain-end-to-end)
below) instead yields **~1564 train events** after `--min-hits` filtering —
a different, smaller draw of STAR events, not a subset of the same one. A
head-to-head check found the qualitative result (pretrained beats
random-init, high significance) holds on both, but the *absolute* silhouette
values shift between them (e.g. ~0.76 vs. ~0.68 pretrained under otherwise
identical settings) — so match `--max-events` to whichever numbers you're
trying to reproduce, don't assume a smaller `--max-events` just gives you a
faster version of the same answer.
**Blah-blah-blah - statistics matter** for benchmarking.

---

## Step 2: Learn STAR's point ordering — `star_voxelizer.py`

**Why this step exists at all:** FM4NPP's backbone is a Mamba (state-space
sequence) model — unlike a plain set-based model, **the order you feed hits
in changes the output**. The paper's real training used a specific ordering
scheme ("Hierarchical Raster Scan": group hits into a 3D eta×phi×radius voxel
grid, visit voxels in a fixed global sweep order, sort hits within a voxel by
radius) driven by precomputed bin-edge statistics. Those bin-edge files were
never published for STAR (or even for reuse — they encode sPHENIX's own
detector geometry). This module recomputes STAR-appropriate bins directly
from STAR data and reconstructs a real, working `Voxelizer` object (FM4NPP's
own class, from `fm4npp.datasets.voxelizer`) around them.

**What it does, concretely:**
- `compute_star_final_bins(eta_n, phi_n, r_raw, r_n)` — quantile-bins eta/phi
  directly from a sample of STAR events (same algorithm as the original);
  detects the inner/outer TPC sector's radial density gap empirically and
  quantile-bins radius within each group (STAR-specific replacement for
  sPHENIX's hardcoded layer thresholds).
- `build_voxelizer(final_bins)` — builds a real `Voxelizer` instance around
  those bins.
- `hierarchical_raster_scan_order(voxelizer, eta_n, phi_n, r_n)` — for one
  event, returns the permutation that puts its hits in the correct order to
  feed to the model.

**Output:** not a file — a fitted `voxelizer` Python object, kept in memory
and reused for every event afterward (fit once on a small held-out set of
"fit" events, e.g. 60, never reused for anything else).

**Touch it yourself** — fit the bins, then watch what the reordering actually
does to one event (color hits by their position in the resulting sequence,
to see the raster-scan sweep pattern directly):

```python
import numpy as np
from mmap_ninja import RaggedMmap
from extract_embeddings_pca import compute_star_stats
from extract_embeddings_linear_probe import event_polar
from star_voxelizer import compute_star_final_bins, build_voxelizer, hierarchical_raster_scan_order

feat = RaggedMmap("data/star_fm4npp_2k/features_train")
stats = compute_star_stats(feat, n_scan=200)

# fit bins on the first 60 events ("fit" events -- never reused downstream)
fit_eta, fit_phi, fit_rraw, fit_rn = [], [], [], []
for i in range(60):
    f = np.asarray(feat[i])
    _, eta_n, phi_n, r_n, r_raw = event_polar(f, stats)
    fit_eta.append(eta_n); fit_phi.append(phi_n); fit_rraw.append(r_raw); fit_rn.append(r_n)
final_bins = compute_star_final_bins(np.concatenate(fit_eta), np.concatenate(fit_phi),
                                      np.concatenate(fit_rraw), np.concatenate(fit_rn))
voxelizer = build_voxelizer(final_bins)

# apply to a fresh event and look at the resulting hit order
f = np.asarray(feat[60])
_, eta_n, phi_n, r_n, _ = event_polar(f, stats)
order = hierarchical_raster_scan_order(voxelizer, eta_n, phi_n, r_n)

import matplotlib.pyplot as plt
plt.scatter(f[order, 1], f[order, 2], c=np.arange(len(order)), cmap="viridis", s=15)
plt.colorbar(label="position in the sequence fed to Mamba")
plt.xlabel("x (cm)"); plt.ylabel("y (cm)")
plt.title("Hierarchical Raster Scan order for one event")
```

You should see the color gradient sweep across the event in voxel order, not
scattered randomly — that's the ordering the sequential model actually
depends on.

**Update — the `dim_sweep_order`/`revert_order` guess has been corrected.**
`build_voxelizer()` takes two more parameters beyond the bins themselves:
`dim_sweep_order`/`revert_order`, which control the traversal order across
the eta/phi/radius voxel grid. This repo originally left them at `(0,1,2)`/
`(0,1,2)` as an unconfirmed guess (no record of the real training config
survives). Reading FM4NPP's own `fm4npp/datasets/dataset_pretrain.py` found
the actual answer: it defines a named-preset `orderdict` (`'EPR'`, `'RPE'`,
`'REP'`, `'PER'`), and **both of FM4NPP's public YAML configs
(`mamba_pretrain.yaml` and `mamba_tracking.yaml`) default to `order: EPR`**,
which maps to `dim_sweep_order=[2,1,0], revert_order=[2,1,0]` — not the
`(0,1,2)`/`(0,1,2)` this repo had been using (that combination turns out to
correspond to the unused `'RPE'` preset). This isn't proof of the exact
`pp_nerf_m3_k30.ckpt` training run specifically (the yaml is anonymized for
publication), but it's FM4NPP's own shipped default, a much stronger prior
than a blind guess.

Both `extract_embeddings_linear_probe.py` and `evaluate_tracking.py` now take
`--dim-sweep-order`/`--revert-order` flags (still defaulting to the old
`(0,1,2)`/`(0,1,2)` for backward compatibility) — pass `--dim-sweep-order
2 1 0 --revert-order 2 1 0` to use FM4NPP's real default instead. Checked
head-to-head (same seed, same everything else): the corrected ordering
**shrinks the pretrained-vs-random-init silhouette gap from +0.078 to +0.052
(about a third smaller) but keeps it highly significant (Wilcoxon p < 0.0001
either way)** — see [README_091626.md](README_091626.md) for the full
comparison table.

---

## Step 3: Frozen embeddings + trained linear probe — `extract_embeddings_linear_probe.py`

**What it does:** for each event, reorders its hits per Step 2, feeds them
through FM4NPP's **frozen** (no gradient updates) pretrained Mamba backbone,
and takes the last layer's per-hit output — a 512-dimensional vector per hit
("embedding"). The paper's own Figure 8 shows these *raw* embeddings don't
separate by track even on the model's native sPHENIX data, so this script
trains one small linear layer (64-d output) on top, using a triplet loss
(pull hits from the same true track together, push hits from different true
tracks apart) on 300 STAR **training** events. It then evaluates **silhouette
score** — how well-separated the true-track groups are in the trained
embedding space — on 100 held-out **test** events never seen during probe
training. The same thing is run once with the pretrained checkpoint and once
with an identical but randomly-initialized backbone, so you get a fair
before/after-pretraining comparison.

**What silhouette score actually measures** (worth being precise about,
since it's easy to over-read): for hit *i*, `s(i) = (b(i) - a(i)) /
max(a(i), b(i))`, where `a(i)` is the mean distance to other hits sharing
*i*'s true track, and `b(i)` is the mean distance to hits of the *nearest
other* true track. It's `+1` if same-track hits are much closer together than
to the next-nearest track, `0` if there's no separation, negative if hits are
actually closer to the wrong track. **It assumes the true grouping is already
known** — it is not clustering, and it does not produce a "reconstructed
track." It only tells you whether the embedding *could plausibly* support
clustering, not whether it actually would.

```bash
python extract_embeddings_linear_probe.py \
    --data-dir data/star_fm4npp_2k \
    --checkpoint checkpoints/pp_nerf_m3_k30.ckpt \
    --n-fit-events 60 --n-train-events 300 --n-test-events 100 \
    --norm star --probe-epochs 30 --seed 1 \
    --dim-sweep-order 2 1 0 --revert-order 2 1 0 \
    --out linear_probe_summary.png
```

(`--dim-sweep-order 2 1 0 --revert-order 2 1 0` matches FM4NPP's own real
default ordering, `order='EPR'` — see the update at the end of Step 2 above.
Omit those two flags to reproduce this repo's earlier numbers under the old,
unconfirmed `(0,1,2)` guess instead.)

**Output:** printed per-model mean±std silhouette and a Wilcoxon p-value on
the paired per-event deltas, plus `linear_probe_summary.png` (boxplots +
delta histogram). Nothing is saved to disk beyond that figure by default —
the embeddings and the trained probe's weights are not persisted.

**Touch it yourself** — pull one event's embedding and compute silhouette by
hand, then look at it in 2D via PCA (this is exactly what `embeddings_pca.png`
shows):

```python
import numpy as np, torch
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from extract_embeddings_linear_probe import build_model, get_frozen_embedding

model = build_model(embed_dim=512, num_layers=12, d_state=32, d_conv=4, expand=2, klen=30,
                     checkpoint_path="checkpoints/pp_nerf_m3_k30.ckpt", load_checkpoint=True)

# E_n, eta_n, phi_n, r_n, order -- from Steps 2/3 above, for one event:
emb = get_frozen_embedding(model, E_n, eta_n, phi_n, r_n, order)   # (N_hits, 512), original hit order

score = silhouette_score(emb, true_track_id)   # true_track_id from seg_target
print("silhouette (raw 512-d embedding):", score)

emb_2d = PCA(n_components=2).fit_transform(emb)
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=true_track_id, cmap="tab10")
plt.title(f"event embedding, PCA-projected (silhouette={score:.3f})")
```

(Run this with `load_checkpoint=False` too, on the same event, to see the
random-init version side by side — that's exactly the comparison
`extract_embeddings_pca.py`'s `--compare-random` flag automates for many
events at once.)

---

## Step 4: From embeddings to hit distributions — `evaluate_tracking.py`

**What it does — and why it had to be added separately:** nothing above ever
assigns a hit to a *predicted* track. Silhouette score only grades an
embedding against the *known* answer. This script adds the missing
clustering step, so you can finally ask "how many hits actually get
reconstructed":

1. **Cluster** each test event's post-probe embeddings with k-means, telling
   it the *true* number of tracks in that event (a deliberate idealization —
   see caveat below).
2. **Match** predicted clusters to true tracks via a Hungarian assignment
   that maximizes total correctly-grouped hits.
3. Per matched pair, compute **efficiency** (matched hits ÷ true track size)
   and **purity** (matched hits ÷ predicted cluster size); a track counts as
   **found** if both are ≥ 50% (the standard "double-majority" convention
   used in HEP tracking / the TrackML challenge).
4. Aggregate: **Adjusted Rand Index** (ARI, a standard clustering-agreement
   score, `1.0` = perfect, `~0` = random) and the **fraction of tracks
   found**, again comparing pretrained vs. random-init.

```bash
python evaluate_tracking.py \
    --data-dir data/star_fm4npp_2k \
    --checkpoint checkpoints/pp_nerf_m3_k30.ckpt \
    --n-fit-events 60 --n-train-events 300 --n-test-events 100 \
    --norm star --probe-epochs 30 --seed 1 \
    --dim-sweep-order 2 1 0 --revert-order 2 1 0 \
    --summary-out tracking_eval_summary.png \
    --events-out tracking_eval_events.png
```

**Output:**
- `tracking_eval_summary.png` — ARI and track-finding-efficiency boxplots,
  pretrained vs. random-init.
- `tracking_eval_events.png` — **the hit distribution plots**: for a handful
  of example test events, every generated (true) hit as a **blue dot** in
  real detector `x`/`y` (cm), with a **red cross** on top of every hit that
  ended up in a track the double-majority rule called "found." A blue dot
  with no red cross on it is a hit that did not get reconstructed.

**Touch it yourself** — reproduce one event's hit-distribution plot directly:

```python
from sklearn.cluster import KMeans
import numpy as np, matplotlib.pyplot as plt

# emb: (N_hits, probe_dim) post-linear-probe embedding for one test event (Step 3)
# true_track_id: (N_hits,) ground truth; feat: (N_hits, 4) = (E, x, y, z)
n_tracks = len(np.unique(true_track_id))
predicted = KMeans(n_clusters=n_tracks, n_init=10).fit_predict(emb)

# (Hungarian-match predicted clusters to true tracks -- see
#  evaluate_tracking.cluster_and_match for the full matching + double-majority logic)

x, y = feat[:, 1], feat[:, 2]
plt.scatter(x, y, c="tab:blue", label="generated hits")
# `correct_mask` marks hits belonging to a found, matched (true track, cluster) pair:
plt.scatter(x[correct_mask], y[correct_mask], marker="x", c="red", label="reconstructed hits")
plt.xlabel("x (cm)"); plt.ylabel("y (cm)"); plt.legend()
```

**Caveat, stated plainly:** giving k-means the *true* track count is an
idealization real track-finding doesn't get. On this STAR sample (mostly 2–3
well-separated tracks per event), it makes clustering close to trivial for
*both* models — which is exactly why this step's ARI/efficiency numbers come
out near-ceiling and don't (yet) show the same pretrained-vs-random gap that
silhouette does. See [README_091626.md](README_091626.md) for the full
discussion of what that does and doesn't mean.

---

## Step 5 (draft, not yet runnable): the paper's real adapter

Steps 3 and 4 are this repo's own lighter stand-ins for the paper's actual
downstream head. FM4NPP's real adapter code exists and is public
(`train/downstream/`) — reading it in detail (`trackinghead.py`, `loss.py`,
`track_finding_trainer.py`) turned up a genuine DETR/Mask2Former-style set
predictor: a fixed number of learned "prototype" queries cross-attend to the
backbone's per-hit features (across ALL 12 layers, weighted-averaged — not
just the last layer, unlike Step 3's probe), each predicting a class
(real-track vs. "no object") and a per-hit mask, trained end-to-end with a
`PointHungarianMatcher` (classification + Dice + Focal cost) — no oracle
track count needed at inference, since unused queries just learn to predict
"no object."

A draft config and supporting scripts exist to try this on STAR with our
checkpoint, but **nothing below has been run end-to-end yet**:

- `configs/star_tracking.yaml` — modeled on FM4NPP's own
  `scripts/configs/mamba_tracking.yaml`, with values overridden for our
  checkpoint's real architecture and STAR's data. Includes one correction to
  an earlier guess in this repo's own history: the number of query
  "prototypes" is **not** a separate hyperparameter — it's literally
  `params.max_gt_classes` in the real trainer code. FM4NPP's public default
  is 150 (sized for busy sPHENIX events); checking STAR's actual per-event
  track count (2000 sampled events) found a max of 4, so the draft config
  uses `max_gt_classes: 10` instead — a data-driven, STAR-specific choice,
  not a paper value.
- `export_voxelizer_bins.py` — fits STAR bins the same way Step 2 does, but
  pickles them to the exact path/filename (`bin_edges_v3_nbins_8_8_6.pkl`)
  FM4NPP's real, *unmodified* `Voxelizer.__init__` expects, so the real
  dataset class can load STAR-fitted bins without any FM4NPP code changes.
  Verified working: produces a correctly-shaped pickle, loadable with plain
  `pickle.load`.
- `convert_star_to_fm4npp.py` (Step 1) now also writes a zero-filled
  `pid_target_{split}/` directory — FM4NPP's real `TPCBatchDataset`
  unconditionally loads one (for an unrelated particle-ID task), even though
  none of this repo's own scripts read it.

**Concrete blockers found, not yet resolved** (tracing the actual CLI
entrypoint, `train_track_finding.py`, and the `train()` method itself — not
just `launch()` — turned up two more beyond the original two):

1. `use_lora: true` imports `fm4npp.models.lora`, which does not exist
   anywhere in the public FM4NPP repo — it would crash. Without LoRA, the
   real trainer's default is to fully fine-tune the backbone (differential
   per-parameter-group learning rates), not keep it frozen like Steps 3–4 —
   a real methodology decision, not a detail.
2. `launch()` calls `restore_checkpoint()` unconditionally, with no config
   flag to skip it for a from-scratch run. **Clarification worth having:**
   FM4NPP's own `--usepretrain`/`--no-pretrain` CLI flag and `train()`'s
   `pretrain=` argument do *not* control this anyway — they control whether
   the downstream head consumes the backbone's per-layer `feature_layers` at
   all, versus bypassing the backbone entirely and processing raw hits
   through its own small embedder. That's a different ablation ("does using
   the backbone help") from the one this whole investigation runs
   throughout ("does *pretraining* the backbone help") — FM4NPP's own yaml
   anticipates our comparison too (`mamba_5m_scratch`, "trains from random
   initialization"), but no CLI-level switch for it currently exists.
3. `train()` unconditionally does `pickle_load(f'{stat_dir}/loss_bin_pp.pkl')`
   and `loss_weight_pp.pkl` — two more of the "never-published" files. These
   look like dead code, though (`self.loss_bin`/`self.loss_weight` are never
   read again anywhere else in the file after being set) — likely fixable
   with two harmless placeholder pickles rather than needing the real files.
4. `train_track_finding.py` itself (the actual CLI script, as opposed to
   `track_finding_trainer.py`) does `params.pretrained_ckpt =
   model2ckpt[args.config]` — it *overrides* whatever's in the YAML with a
   lookup in a hardcoded dict of the original authors' own config names and
   filesystem paths. Our config name isn't in that dict → immediate
   `KeyError`. Easiest fix: don't use that script at all — write our own
   small launcher in this repo that imports `DownstreamTrainer` directly and
   calls `.launch()` + `.train()` using our YAML's `pretrained_ckpt` as-is,
   without touching FM4NPP's own files.

Net effect: (1) and (3) look cheap to route around; (2) is a real decision
requiring actual new code (not just a config value) to get a genuine
pretrained-vs-random-backbone comparison out of this trainer; (4) is avoided
entirely by not using the stock CLI script. None of this has been attempted
yet — flagged here rather than run, pending a decision on whether it's worth
the (nontrivial, multi-epoch-training-scale) effort.

**Update: sidestepped rather than resolved.** Instead of fixing the four
blockers above in FM4NPP's own CLI scripts (`train_track_finding.py`,
`DownstreamTrainer.launch()`/`.train()`), a separate, much smaller script
(`eval_track_finding_star.py`) imports the *real model code* those scripts
also use — `trackinghead.MambaAttentionHead`, `loss.PointHungarianMatcher`,
`loss.compute_point_loss`, `loss.assign_points_to_masks`, all unmodified —
and wraps it in its own minimal training/eval loop, the same pattern Steps
3–4 already use around `MambaGPT`. This avoids all four blockers rather than
fixing them: no `use_lora`, no `launch()`/`restore_checkpoint()`, no
`stat_dir` pickle loads, no hardcoded checkpoint dict. See Step 6 below for
what this actually found.

---

## Step 6: the paper's real adapter — actually run, multi-seed checked

`eval_track_finding_star.py` is the STAR-adapted mimic of FM4NPP's real
`train/downstream/eval_track_finding.py` workflow described in Step 5 above,
built by directly importing the paper's real downstream-adapter code rather
than trying to unblock its own CLI scripts. Per event:

1. Get **all 12 backbone layers'** frozen features (not just the last one,
   unlike Step 3) — `feature = torch.stack(pre_embed)`, matching exactly what
   `track_finding_trainer.py` feeds its head.
2. Feed them into a real `MambaAttentionHead`: a DETR/Mask2Former-style set
   predictor with `num_prototypes` learned query embeddings that cross-attend
   to the hits and each predict a class (real-track vs. "no object") and a
   per-hit mask.
3. Train it with the real `PointHungarianMatcher` + `compute_point_loss`
   (classification + Dice + focal losses), on STAR train events, with the
   backbone kept frozen throughout (same policy as Steps 3–4).
4. Decode with the real `assign_points_to_masks()`: each hit goes to
   whichever query scores highest, or is left unassigned if every query says
   "no object." **No track count is ever given to the model** — unlike Step
   4's k-means, which is handed the true count as an oracle input.

```python
# feat_stack: (12, N_hits, 512) frozen backbone features for one event, ALL layers, original hit order
# lab: (N_hits,) true track id -- from Steps 1-3 above
outputs, padding_mask = forward_event(down_model, feat_stack, device)          # class_probs, mask_probs
decoded = assign_points_to_masks(outputs, no_object_class=0, option=2, threshold=0.0)
assignments = decoded["assignments"][0].cpu().numpy()   # (N_hits,) predicted track id per hit, -1 = unassigned
ari = adjusted_rand_score(lab, assignments)              # same metric Step 4 reports, oracle-free this time
```

**Single-seed result** (seed 42, the default; 60 fit / 300 train / 100 test
events, 30 head-training epochs, backbone frozen):

| metric | pretrained | random-init | delta | Wilcoxon p |
|---|---|---|---|---|
| ARI | 0.896 ± 0.271 | 0.795 ± 0.333 | +0.100 | 0.0004 |
| frac_found (double-majority) | 0.923 ± 0.167 | 0.903 ± 0.211 | +0.020 | 0.206 |

Taken alone, this looks like exactly the confirmation Step 4's ceiling-effect
diagnosis predicted: remove the oracle track count, and a real, significant
ARI gap reappears. **But treat that as provisional until multi-seed-checked
— which, for this method, changes the picture:**

**Multi-seed check** (5 seeds, identical settings otherwise):

| seed | pretrained ARI | random-init ARI | delta | Wilcoxon p |
|---|---|---|---|---|
| 42 | 0.896 ± 0.271 | 0.795 ± 0.333 | +0.1001 | 0.0004 |
| 1 | 0.922 ± 0.231 | 0.913 ± 0.246 | +0.0090 | 0.6095 |
| 2 | 0.896 ± 0.261 | 0.925 ± 0.218 | **−0.0297** | 0.1126 |
| 3 | 0.910 ± 0.252 | 0.834 ± 0.314 | +0.0759 | 0.0079 |
| 4 | 0.870 ± 0.297 | 0.903 ± 0.254 | **−0.0333** | 0.3122 |

Mean of the 5 per-seed deltas: **+0.0244** (std across seeds ≈ **0.0545**) —
only 2 of 5 seeds individually significant at p<0.05, and the sign flips
negative in 2 of 5. (frac_found across the same 5 seeds: mean delta
+0.0035, std 0.0116 — essentially no effect, none individually significant —
consistent with Step 4's finding for that same coarser metric.) With only 5
seeds this repo's own convention (see README_082026.md) is to not run a
seed-level significance test at all — n=5 is too small either way.

**Correct framing, not overstated in either direction:** the single-seed
run's clean p=0.0004 result does **not** replicate robustly — this is a
different outcome from the silhouette probe's multi-seed check
(README_082026.md), where all 5 seeds were positive and significant. The
honest read here is *"weak, seed-dependent evidence of a positive effect on
ARI under the real, non-oracle adapter — clearly stronger and more often
positive than Step 4's near-null oracle-k result, but not itself a settled,
robust finding."* Don't report the seed-42 table's p=0.0004 on its own; don't
report "no effect" either — both would overstate what 5 noisy, mixed-sign
seeds actually support.

**A plausible (not paper-stated) reason this is noisier than the silhouette
probe:** `MambaAttentionHead` has ~3.9M trainable parameters, trained on only
300 events for 30 epochs — several orders of magnitude more capacity, on
much less data, than the linear probe's ~33k parameters. Head-training
variance (a different, freshly-initialized head is trained per arm per seed)
is plausibly a bigger source of run-to-run noise here than backbone identity
itself, on top of whatever real transfer effect exists.

**Same standing caveats as Steps 3–5, plus one specific to this method:**
`num_prototypes=10` and the loss/matcher cost weights are FM4NPP's public
defaults, not STAR-tuned; the "random-init" arm re-randomizes the whole
backbone (still not what FM4NPP's own `--usepretrain=False` actually does,
per Step 5's blocker 2); and the regression/PID/noise auxiliary heads
`MambaAttentionHead` defines are unused (no truth available for them in the
STAR conversion).

---

## How this lines up with the FM4NPP paper

| This repo's step | Paper's methodology | Status |
|---|---|---|
| `(E, x, y, z)` per-hit feature (Step 1) | Real per-hit input feature (verified from the checkpoint's own code, not from FM4NPP's own — incorrect — docs) | **Exact match.** |
| Frozen `MambaGPT` backbone, `pp_nerf_m3_k30.ckpt` (Step 3) | The actual pretrained Mamba2 backbone, self-supervised on TPCpp-10M (sPHENIX) | **Exact match** — this repo reuses FM4NPP's real released weights and architecture unmodified. |
| Polar transform + normalize (`event_polar`) | Paper's `(E,x,y,z) → (E,eta,phi,r) → normalize` preprocessing | **Exact match** in formula; STAR-specific normalization constants recomputed from data (`--norm star`) rather than reused from sPHENIX. |
| `star_voxelizer.py`'s eta/phi quantile binning (Step 2) | Paper's Hierarchical Raster Scan `Voxelizer.equalObs()` binning | **Same algorithm**, refit on STAR data since the original sPHENIX bin-edge file was never published. |
| `star_voxelizer.py`'s radius binning (Step 2) | Paper's hardcoded sPHENIX layer-group thresholds (40/57 cm) | **Deliberate substitution** — STAR's TPC has different physical layer geometry, so this repo detects the radial density gap empirically instead. |
| `Voxelizer`'s `dim_sweep_order`/`revert_order` (Step 2) | Part of the real training config | **Corrected, not just flagged.** FM4NPP's own public configs both default to `order='EPR'` → `[2,1,0]/[2,1,0]`, not this repo's original `(0,1,2)`/`(0,1,2)` guess. Verified effect: shrinks the pretrained-vs-random silhouette gap from +0.078 to +0.052, still p<0.0001. Still not a guarantee of the exact checkpoint's real training config (yaml is anonymized for publication). |
| Linear probe + triplet loss (Step 3) | Paper's real downstream adapter: a transformer-decoder instance-segmentation head trained with Hungarian matching + Dice/Focal/classification losses (paper Fig. 4) | **Lighter stand-in**, motivated directly by the paper's own Fig. 8 finding that a linear projection is the minimum needed to see any separation at all. Not a reimplementation of the real adapter — see Step 5 for a drafted (not yet run) path toward the real one. |
| Silhouette score (Step 3) | Not in the paper | **This repo's own addition** — a cheap transfer-learning probe, not a paper-reported metric. |
| k-means + Hungarian double-majority (Step 4) | Paper reports ARI and segmentation efficiency from its *real* adapter's output on sPHENIX | **Same metric names, different machinery** — this repo's numbers use oracle-k clustering on top of the lighter linear probe, not the paper's learned segmentation head, so they are not directly comparable to the paper's own reported ARI/efficiency values. Directionally analogous only. |
| Real transformer-decoder adapter (Step 5 draft; Step 6 run) | The actual thing (paper Fig. 4) | **Run, not just drafted** — but via a standalone script that directly imports the paper's real `MambaAttentionHead`/`PointHungarianMatcher`/`assign_points_to_masks` code, sidestepping (not fixing) the four CLI blockers Step 5 found. Single-seed result looked like a clean, significant pretrained-vs-random-init ARI gap (p=0.0004); the 5-seed check found that doesn't replicate robustly (mean delta +0.024, only 2/5 seeds individually significant, sign flips in 2/5) — see Step 6. |

The short version: Steps 1 and 3's *backbone* are a faithful, code-verified
reuse of the paper's actual released model and preprocessing. Step 2 fills in
a genuine gap (missing bin-edge files) with the same algorithm the paper
describes, on STAR-specific data, and its remaining ordering assumption is
now corrected against FM4NPP's own public default, not just flagged as
unknown. Step 3's *probe* and Step 4's *clustering* are this repo's own,
deliberately lighter-weight stand-ins for the paper's real downstream
adapter — built to test *whether* transfer exists cheaply, not to reproduce
the paper's reported numbers. Step 6 closes that last gap with the paper's
actual adapter code (Step 5 documents the CLI path that would have been
needed to run FM4NPP's own scripts unmodified, and why that was blocked) —
its multi-seed result is real but weaker and noisier than the single-seed
run first suggested, not a clean confirmation.

---

## Quick reference: running the full chain end-to-end

Once you understand *why* each step exists (above), here are the commands
back-to-back, assuming a STAR simulation ROOT file already in the correct
branch format and the environment already set up:

```bash
source ~/fm4npp-env/bin/activate
export PYTHONPATH=/home/daq/Work/STAR/PP_collision:$PYTHONPATH
cd /home/daq/Work/STAR/fm4npp4starTPC

# Step 1 -- convert
python convert_star_to_fm4npp.py \
    --input data/merged_TPCHitsTree.root --tree T \
    --output-dir data/star_fm4npp_2k \
    --max-events 2000 --train-frac 0.8 --overwrite

# Step 3 -- silhouette probe (the headline result)
python extract_embeddings_linear_probe.py \
    --data-dir data/star_fm4npp_2k \
    --checkpoint checkpoints/pp_nerf_m3_k30.ckpt \
    --n-fit-events 60 --n-train-events 300 --n-test-events 100 \
    --norm star --probe-epochs 30 --seed 1 \
    --dim-sweep-order 2 1 0 --revert-order 2 1 0 \
    --out linear_probe_summary.png

# Step 4 -- tracking-level evaluation + hit-distribution plots
python evaluate_tracking.py \
    --data-dir data/star_fm4npp_2k \
    --checkpoint checkpoints/pp_nerf_m3_k30.ckpt \
    --n-fit-events 60 --n-train-events 300 --n-test-events 100 \
    --norm star --probe-epochs 30 --seed 1 \
    --dim-sweep-order 2 1 0 --revert-order 2 1 0 \
    --n-plot-events 6 \
    --summary-out tracking_eval_summary.png \
    --events-out tracking_eval_events.png

# Step 6 -- real, oracle-free adapter (MambaAttentionHead), trained + evaluated on STAR
python eval_track_finding_star.py \
    --data-dir data/star_fm4npp_2k \
    --checkpoint checkpoints/pp_nerf_m3_k30.ckpt \
    --n-fit-events 60 --n-train-events 300 --n-test-events 100 \
    --head-epochs 30 --seed 42 \
    --out track_finding_head_summary.png
```

Read the printed `Wilcoxon p=` lines before trusting any result, and rerun
across a few `--seed` values before treating any single run's delta as
settled — this mattered for Step 3 (5/5 seeds positive, confirmed robust,
see README_082026.md) and mattered in the *opposite* direction for Step 6
(single-seed p=0.0004 did not replicate across 5 seeds, see Step 6 above and
README_091626.md). Full explanation of what each output means and how much
to trust it: [README_082026.md](README_082026.md) (silhouette result) and
[README_091626.md](README_091626.md) (tracking-level result, its
ceiling-effect caveat, and the real-adapter multi-seed check).
