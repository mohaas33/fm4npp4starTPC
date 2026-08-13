# STAR TPC × FM4NPP: Cross-Detector Transfer Probe

This repo applies [FM4NPP](https://github.com/FM4NPP/PP_collision) — a Mamba-based
foundation model pretrained on 11M simulated **sPHENIX** TPC p+p collision events
— to **STAR** TPC simulation data, to test whether the pretrained representations
carry any detectable, transferable physics structure across detectors.

FM4NPP was never trained on STAR data or STAR geometry. This is a cross-detector
transfer-learning experiment, not a "just run the pretrained model" exercise.

## Status: negative result so far, at the frozen-embedding stage

**TL;DR:** a frozen (untrained) FM4NPP backbone, fed STAR hits under two different
normalization schemes, shows **no detectable advantage over a randomly initialized
model of the same architecture** at separating hits by true track ID (n=100 STAR
events, mean pretrained-vs-random delta ≈ +0.007, distribution straddles zero,
not statistically significant). See [Results](#results) below for the full picture
and — importantly — why this doesn't settle the question. The likely next step
(training a LoRA adapter on the frozen backbone) is described in
[Next Steps](#next-steps).

## Quick test

Two commands to check the pipeline actually works on your machine, once
[Environment setup](#environment-setup) is done.

**Test 1 — convert STAR data → FM4NPP format.** Uses the small real-data
fixture committed at `tests/fixtures/test_100events.root` (100 events,
~170 KB), so this works right after cloning — no download needed:

```bash
python convert_star_to_fm4npp.py \
    --input tests/fixtures/test_100events.root \
    --tree T \
    --output-dir /data/test_conversion_check \
    --max-events 100 \
    --overwrite
```

Expect output ending in `Kept 99 events` (one fixture event genuinely has 0
hits, same as in the full source file) and `[DONE] Conversion complete.`
Swap `/data/test_conversion_check` for any writable directory.

**Test 2 — frozen-backbone embedding probe.** This one needs the pretrained
checkpoint and a full converted STAR dataset of your own — see
[Getting the checkpoint and input data](#getting-the-checkpoint-and-input-data)
and [Usage](#usage) below to produce `data/star_fm4npp_2k/`:

```bash
python extract_embeddings_pca.py \
    --data-dir data/star_fm4npp_2k \
    --checkpoint checkpoints/pp_nerf_m3_k30.ckpt \
    --split train \
    --n-events 100 \
    --min-hits 15 \
    --norm star \
    --compare-random \
    --out embeddings_pca.png \
    --summary-out silhouette_summary.png
```

Expect a `[RESULT] event ...: silhouette=...` line per event, then a
`=== SUMMARY ===` block with mean silhouette scores and a Wilcoxon/t-test
p-value (should roughly match [Results](#results) below), plus two PNGs
written to the current directory.

## Repository contents

```
.
├── convert_star_to_fm4npp.py   # STAR ROOT TTree -> FM4NPP RaggedMmap format
├── extract_embeddings_pca.py   # frozen-backbone embedding probe + PCA + stats
├── extract_test_fixture.py     # regenerates tests/fixtures/test_100events.root from the real source file
├── test_convertion.py          # sanity checks on converted data
├── tests/
│   └── fixtures/
│       └── test_100events.root # committed: real 100-event subset, for fast offline pipeline testing
├── checkpoints/
│   └── pp_nerf_m3_k30.ckpt     # pretrained FM4NPP checkpoint (NOT committed to git — see below)
└── data/
    ├── merged_TPCHitsTree.root # source STAR TPC simulation (NOT committed — see below)
    └── star_fm4npp_2k/         # converted RaggedMmap output (NOT committed — see below)
```

**Do not commit `checkpoints/` or `data/` to git.** The checkpoint is tens of MB and
the ROOT/RaggedMmap files can be much larger; none of it belongs in version control.
This repo's `.gitignore`:

```gitignore
checkpoints/
data/
*.root
!tests/fixtures/*.root
*.ckpt
__pycache__/
*.pyc
```

The one deliberate exception is `tests/fixtures/*.root`: a small (~170 KB), real
100-event subset of `data/merged_TPCHitsTree.root` (not synthetic — a straight,
unmodified extraction of the tree's first 100 entries, every branch, same values
and jagged structure as the source), committed to git so the conversion pipeline
can be tested and exercised without the full ~12GB source file. See
[Test fixture](#test-fixture) below.

Document where to *get* the checkpoint and full input data instead (see below), so
the repo is reproducible without bloating it.

## Background

[FM4NPP](https://github.com/FM4NPP/PP_collision) is a Mamba/Mamba2 state-space
foundation model, self-supervised pretrained on the TPCpp-10M dataset (sPHENIX TPC
p+p collisions). This repo asks: does anything learned from sPHENIX geometry
transfer to STAR, a physically different TPC (larger radius, different pad/sector
layout, different readout)?

Two important repo-level caveats worth stating up front, discovered while getting
this working — the upstream FM4NPP repo's documentation (`README.md`, `SETUP.md`,
`example_usage.py`) is **inconsistent with its actual code** in several places:

- The real per-hit feature vector is 4D `(E, x, y, z)`, not the 30D vector described
  in `SETUP.md`/`example_usage.py`.
- The real pretrained checkpoint architecture is FM4NPP's own custom `Mamba2`
  (`fm4npp/models/mamba2.py`, exposed as the `MambaGPT` class), not `Mamba1GPT`, and
  not a `Mamba2GPT` class (which doesn't exist in the shipped code at all).
- `example_usage.py` does not run as-is (bad import, invalid `embed_method` value,
  wrong input tensor axis order).
- The `Voxelizer`/space-filling-curve point-ordering stat files
  (`bin_edges_v3_nbins_8_8_6.pkl`, `loss_bin_pp.pkl`, `loss_weight_pp.pkl`) referenced
  in the training config are **not included** in the public repo or the checkpoint
  Google Drive folder. This repo's embedding extraction therefore uses plain
  r-sorted points instead of the real voxel-serialized order (see
  [Known Limitations](#known-limitations)).

Everything in this repo was derived by reading FM4NPP's actual source
(`fm4npp/models/`, `fm4npp/datasets/`, `train/downstream/`), not by trusting its
docs — worth keeping in mind if you extend this further.

## Environment setup

This needs a real NVIDIA GPU on Linux (CUDA-compiled kernels, no CPU/Mac path).
Tested on Ubuntu 22.04.2, RTX 3070 (8GB VRAM), CUDA 12.1.

```bash
python3.10 -m venv ~/fm4npp-env
source ~/fm4npp-env/bin/activate
pip install --upgrade pip

pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121

# causal-conv1d: build from git tag v1.4.0 specifically.
# Neither PyPI (missing csrc/ source) nor the latest release (incompatible
# in-place API vs. what mamba-ssm 2.2.2 expects) work here.
git clone --depth 1 --branch v1.4.0 https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d && CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=2 \
    pip install . --no-build-isolation --no-cache-dir --no-deps && cd ..

# mamba-ssm: build from git tag v2.2.2 specifically.
# PyPI sdist is missing csrc/ source; latest release pulls in Mamba-3/tilelang/
# cutlass-dsl deps that silently upgrade torch and break this pinned setup.
git clone --depth 1 --branch v2.2.2 https://github.com/state-spaces/mamba.git
cd mamba && MAMBA_FORCE_BUILD=TRUE MAX_JOBS=2 \
    pip install . --no-build-isolation --no-cache-dir --no-deps && cd ..

pip install triton mmap-ninja pyyaml numpy scipy tqdm scikit-learn matplotlib uproot awkward
```

**One required manual patch:** `mamba_ssm/__init__.py` unconditionally imports
`MambaLMHeadModel`, which pulls in a `transformers.generation` symbol
(`GreedySearchDecoderOnlyOutput`) that newer `transformers` releases removed. FM4NPP
never uses `MambaLMHeadModel`, so comment that one import line out in your installed
package:

```bash
MAMBA_INIT="$(python -c "import mamba_ssm.__file__ if False else None" 2>/dev/null; \
  find "$(python -c 'import sys; print(sys.prefix)')" -path '*/mamba_ssm/__init__.py')"
sed -i 's/^from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel/# &/' "$MAMBA_INIT"
```
(Find the file manually via `pip show -f mamba-ssm` if the one-liner above doesn't
resolve on your setup — the import fails before you can `import mamba_ssm` normally
to introspect its own path, so avoid any approach that requires importing it first.)

Verify everything:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from mamba_ssm import Mamba; print('mamba-ssm OK')"
python -c "import causal_conv1d; print('causal-conv1d OK')"
```

Clone FM4NPP itself alongside this repo and put it on your `PYTHONPATH`:
```bash
git clone https://github.com/FM4NPP/PP_collision.git
export PYTHONPATH=/path/to/PP_collision:$PYTHONPATH
```

## Getting the checkpoint and input data

- **Pretrained checkpoint**: from FM4NPP's README, the "Model Checkpoints" Google
  Drive link. This repo was built and tested against `pp_nerf_m3_k30.ckpt`
  (`embed_dim=512, num_layers=12, d_state=32, headdim=64, ngroups=1, klen=30`,
  `pe_method='nerf'`, FM4NPP's custom Mamba2 block — architecture derived directly
  from the checkpoint's own `state_dict` tensor shapes, since the filename's "m3"
  does not match the paper's own Table 1 model-size naming for that width).
- **STAR TPC simulation data**: a ROOT TTree (`merged_TPCHitsTree.root` here) with
  branches `TPCHits_x/y/z/q/adc/pad/row/sector/timebucket/IdTruth` per hit, one
  entry per event. Not included in this repo — supply your own STAR simulation
  output with this branch structure, or adapt `convert_star_to_fm4npp.py`'s
  `--x-branch`/`--y-branch`/etc. flags to your actual branch names.

## Usage

### 1. Convert STAR ROOT data to FM4NPP's RaggedMmap format

```bash
python convert_star_to_fm4npp.py \
    --input data/merged_TPCHitsTree.root \
    --tree T \
    --output-dir data/star_fm4npp_2k \
    --max-events 2000 \
    --train-frac 0.8 \
    --overwrite
```

Builds `(E, x, y, z)` per hit (`E` from `TPCHits_adc`, matching the ADC-signal
convention FM4NPP's `E` feature uses for sPHENIX) and `track_id` from
`TPCHits_IdTruth`. Writes `features_{train,test}/`, `seg_target_{train,test}/`, and
a zero-filled placeholder `reg_target_{train,test}/` (required by FM4NPP's dataset
loader but unused for anything in this repo — see the script's docstring).

Sanity-check the output with `test_convertion.py` before moving on.

### Test fixture

`tests/fixtures/test_100events.root` is a small, real 100-event subset of
`data/merged_TPCHitsTree.root` — the source file's first 100 entries, every
branch, unmodified — committed to git (see [Quick test](#quick-test) above to
run the conversion pipeline against it right away, no download needed).
Hits/event over the 99 non-empty events: min=3, max=116, mean≈52.

To regenerate the fixture from a fresh copy of the source file (e.g. after a
resimulation, or to pull a different/larger slice):

```bash
python extract_test_fixture.py \
    --input data/merged_TPCHitsTree.root \
    --tree T \
    --n-events 100 \
    --output tests/fixtures/test_100events.root \
    --overwrite
```

Verified readable both by `convert_star_to_fm4npp.py` (uproot) and by a real
`root` CLI/PyROOT install (`root -l tests/fixtures/test_100events.root`,
`T->Show(0)`). Note for anyone touching `extract_test_fixture.py`: as of
uproot 5.7.0, plain dict-assignment (`file[name] = {...}`) defaults to writing
an **RNTuple**, not a TTree — unreadable by a stock `root`/PyROOT install and
a silent behavior change from older uproot. The script uses
`file.mktree(name, data)` explicitly to get a real TTree instead.

### 2. Frozen-backbone embedding probe

```bash
python extract_embeddings_pca.py \
    --data-dir data/star_fm4npp_2k \
    --checkpoint checkpoints/pp_nerf_m3_k30.ckpt \
    --split train \
    --n-events 100 \
    --min-hits 15 \
    --norm star \
    --compare-random \
    --out embeddings_pca.png \
    --summary-out silhouette_summary.png
```

Replicates FM4NPP's real preprocessing (`(E,x,y,z) → polar (E,eta,phi,r) →
normalize → sort by r`), minus the `Voxelizer` serialization step (see
[Known Limitations](#known-limitations)). Extracts last-layer per-point embeddings
from the frozen pretrained backbone, and — with `--compare-random` — from an
identically-shaped but randomly initialized control model, on the *same* events.
Reports a per-event silhouette score (true track ID as the label) for both, plus a
paired Wilcoxon/t-test on the deltas.

`--norm star` recomputes eta/phi/r/E normalization stats from your STAR sample;
`--norm sphenix` keeps FM4NPP's original hardcoded sPHENIX constants, for a
true-zero-shot comparison.

## Results

Run at `n=100` STAR events, `min-hits=15`, both normalization modes:

| norm mode | mean silhouette (pretrained) | mean silhouette (random-init) | mean delta |
|---|---|---|---|
| `star`    | ~0.20 | ~0.19 | +0.0074 |
| `sphenix` | ~0.19 | ~0.19 | +0.0076 |

Both normalization modes give essentially the same near-zero delta, with the
per-event delta histogram straddling zero in both cases (roughly symmetric mass on
either side, no statistically significant shift). See `silhouette_summary.png` for
the full distribution and `embeddings_pca.png` for qualitative per-event PCA plots.

**Interpretation:** at the frozen, zero-shot embedding stage, this probe finds no
detectable advantage of the pretrained weights over random initialization for
separating STAR hits by track. Critically, the fact that both `star` and `sphenix`
normalization give the *same* null result rules out normalization/domain-gap
mismatch as the explanation — pointing instead at the point-ordering issue below as
the more likely cause of the null result.

## Known limitations

- **Missing Voxelizer stat files.** FM4NPP's real data pipeline reorders points
  via a space-filling-curve/voxel-grouping step before feeding them to the
  (sequential, Mamba-based) model, using bin-edge statistics
  (`bin_edges_v3_nbins_8_8_6.pkl`, `loss_bin_pp.pkl`, `loss_weight_pp.pkl`) that are
  **not included** in the public FM4NPP repo or checkpoint download. This repo
  substitutes plain r-sorted ordering instead — a genuine code path in FM4NPP's own
  dataset class, but not what the model was actually trained on. Since Mamba is
  order-sensitive, this is the leading candidate explanation for the null result
  above, and should be resolved (e.g. by requesting the missing files from the
  FM4NPP authors) before treating the negative result as final.
- **`reg_target` is a zero-filled placeholder.** The STAR hit tree used here has no
  per-hit truth momentum/vertex information, so the auxiliary regression target
  FM4NPP's dataset loader expects is filled with zeros. This only matters if you
  train with `return_reg=True`; nothing in this repo currently does.
- **Small/imbalanced event samples.** Some STAR events have very few hits or only
  a single track, which silhouette score can't meaningfully evaluate (reported as
  `NaN` and excluded from aggregate statistics).

## Next steps

The frozen-embedding probe is a cheap first filter, not a final answer — especially
given the point-ordering limitation above. The more decisive experiment: freeze the
pretrained FM4NPP backbone, train a lightweight LoRA adapter + track-finding head on
top of it using STAR data, and compare against the same adapter/head trained on a
randomly initialized (from-scratch) backbone. A trainable adapter can compensate for
ordering/domain mismatches that a purely frozen forward pass cannot, making it a
fairer test of whether the pretrained weights carry transferable structure. This
uses FM4NPP's own `train/downstream/track_finding_trainer.py` with
`mambaversion: mamba2`, `pretrained_ckpt` pointing at the checkpoint above, and
`use_lora: true`.

## Acknowledgments

Built on top of [FM4NPP/PP_collision](https://github.com/FM4NPP/PP_collision).
See that repo for the original pretraining methodology and the TPCpp-10M dataset
paper. This repo is an independent cross-detector transfer experiment and is not
affiliated with the original FM4NPP authors.