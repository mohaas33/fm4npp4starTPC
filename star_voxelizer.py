#!/usr/bin/env python3
"""
STAR-adapted Hierarchical Raster Scan serialization.

FM4NPP's real training-time point ordering (paper: "Hierarchical Raster Scan",
Appendix Methodology/Serialization) is implemented by fm4npp.datasets.voxelizer
.Voxelizer, but the pretrained checkpoint's precomputed bin-edge .pkl files
were never published. Reading Voxelizer.__init__ shows this was never a hard
requirement -- it self-computes bins from data when the .pkl is absent:

    if os.path.exists(bin_addr):
        self.final_bins = self.pickle_load(bin_addr)
    else:
        assert dataset is not None, 'dataset must be provided for computing bins'
        self.get_data(dataset, limit=limit)
        self.final_bins = self.compute_bins(n_bins=n_bins)

Two of its three binning rules are directly reusable for STAR as-is:
    - Eta bins: quantile ("equalObs") binning on normalized eta values.
    - Phi bins: quantile ("equalObs") binning on normalized phi values.
Both are computed straight from data with no sPHENIX-specific constants.

The third rule (Voxelizer.get_Radius_bins) is NOT reusable as-is: it hardcodes
sPHENIX's own detector geometry (layer_split_thresholds=[40, 57] cm, reflecting
"the detector's 48 layers are arranged in three major groups" per the paper,
plus sPHENIX-specific r_normalize() min/max constants). STAR's TPC has a
different physical layer structure, so this module reimplements radius binning
using a data-driven density-gap detection instead of a hardcoded sPHENIX
literature value -- see detect_radial_groups() below.

Rather than fighting Voxelizer's __init__/get_data() (which expects a full
TPCBatchDataset + DataLoader + collate_fn), this module computes a
STAR-appropriate `final_bins` dict directly and constructs a working Voxelizer
instance by bypassing __init__ and setting state manually -- Voxelizer's own
tokenize()/grouping() methods only need self.final_bins and self.refarray to
work correctly, both of which are set here identically to how __init__ would
set them from a loaded/pickled bins dict.
"""

import numpy as np
import torch


def equal_obs_bins(x, nbin):
    """
    Quantile (equal-count) bin edges. Identical algorithm to Voxelizer's own
    equalObs() in fm4npp/datasets/voxelizer.py -- reimplemented here so this
    module has no import-time dependency on that file (which pulls in plotly/
    matplotlib/seaborn at module load, unnecessary for headless bin computation).
    """
    x = np.asarray(x)
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1), np.arange(nlen), np.sort(x))


def detect_radial_groups(r_raw, n_groups=2):
    """
    Data-driven replacement for Voxelizer.get_Radius_bins()'s hardcoded
    sPHENIX layer_split_thresholds=[40, 57]. Detects natural density valleys
    in the RAW (un-normalized, cm) radial distribution -- STAR's inner/outer
    TPC sector transition should show up as a genuine drop in hit density,
    the same physical signature the original code exploited for sPHENIX's
    3-group layer structure, just found empirically instead of hardcoded.

    Returns n_groups-1 boundary values in raw r (cm), sorted ascending.
    """
    r_raw = np.asarray(r_raw)
    hist, edges = np.histogram(r_raw, bins=200)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Smooth lightly to avoid picking noise spikes as "valleys"
    kernel = np.ones(5) / 5
    smoothed = np.convolve(hist, kernel, mode='same')

    # Find the n_groups-1 deepest local minima that actually separate real mass
    # on both sides (avoids picking edge-of-distribution noise as a "valley").
    from scipy.signal import argrelextrema
    minima_idx = argrelextrema(smoothed, np.less_equal, order=5)[0]
    minima_idx = [i for i in minima_idx if 0 < i < len(smoothed) - 1]

    if len(minima_idx) < n_groups - 1:
        print(f"[WARN] detect_radial_groups: found only {len(minima_idx)} candidate valleys, "
              f"need {n_groups - 1}. Falling back to equal-quantile radial split -- "
              f"this is a weaker substitute for a real physical boundary, worth "
              f"double-checking against STAR's actual padrow geometry if available.")
        boundaries = list(np.quantile(r_raw, np.linspace(0, 1, n_groups + 1)[1:-1]))
        return boundaries

    # rank candidate valleys by how deep they are relative to local neighborhood
    depths = [(smoothed[i], i) for i in minima_idx]
    depths.sort()
    chosen = sorted([bin_centers[i] for _, i in depths[:n_groups - 1]])
    return chosen


def compute_star_final_bins(eta_n, phi_n, r_raw, r_n, n_bins=(8, 8, 6), n_radial_groups=2):
    """
    Build the {'Eta': [...], 'Phi': [...], 'Radius': [...]} dict Voxelizer
    expects, computed entirely from STAR data.

    eta_n, phi_n, r_n : normalized (0-1) arrays -- same normalization your
        polar_and_normalize() already applies (either 'star' or 'sphenix' mode).
        Bin edges are computed in this normalized space, matching how the
        original Voxelizer.compute_bins() operates (final_bins values are used
        directly against normalized coordinates in tokenize()).
    r_raw : the RAW (un-normalized, cm) radius array, used only for detecting
        the physical inner/outer sector density valley -- normalization would
        obscure the actual physical gap.
    n_bins : (eta_bins, phi_bins, radius_bins), matching Voxelizer's n_bins
        ordering convention.
    n_radial_groups : how many physically-motivated radial groups to split
        into before sub-binning each with equal_obs_bins. STAR's TPC has a
        2-group inner/outer sector structure (unlike sPHENIX's 3), so this
        defaults to 2 -- override if you have reason to believe otherwise.
    """
    final_bins = {}

    eta_edges = list(equal_obs_bins(eta_n, n_bins[0]))
    eta_edges[0], eta_edges[-1] = 0.0, 1.0
    final_bins['Eta'] = eta_edges

    phi_edges = list(equal_obs_bins(phi_n, n_bins[1]))
    phi_edges[0], phi_edges[-1] = 0.0, 1.0
    final_bins['Phi'] = phi_edges

    n_r_bins = n_bins[2]
    boundaries_raw = detect_radial_groups(r_raw, n_groups=n_radial_groups)
    print(f"[INFO] Detected {len(boundaries_raw)} radial group boundary(ies) at raw r = "
          f"{[f'{b:.1f}cm' for b in boundaries_raw]}")

    # map raw boundaries into the same normalized r space as r_n, via the
    # empirical relationship between r_raw and r_n over this sample
    order = np.argsort(r_raw)
    r_raw_sorted, r_n_sorted = np.asarray(r_raw)[order], np.asarray(r_n)[order]
    boundaries_n = list(np.interp(boundaries_raw, r_raw_sorted, r_n_sorted))
    r_min_n, r_max_n = float(np.min(r_n)), float(np.max(r_n))
    group_edges_n = [r_min_n] + boundaries_n + [r_max_n]

    bins_per_group = [n_r_bins // n_radial_groups] * n_radial_groups
    for i in range(n_r_bins - sum(bins_per_group)):  # distribute any remainder
        bins_per_group[i] += 1

    radius_edges = [group_edges_n[0]]
    for g in range(n_radial_groups):
        lo, hi = group_edges_n[g], group_edges_n[g + 1]
        mask = (r_n >= lo) & (r_n <= hi)
        if mask.sum() < bins_per_group[g] + 1:
            sub_edges = np.linspace(lo, hi, bins_per_group[g] + 1)
        else:
            sub_edges = equal_obs_bins(r_n[mask], bins_per_group[g])
            sub_edges[0], sub_edges[-1] = lo, hi
        radius_edges.extend(list(sub_edges[1:]))
    final_bins['Radius'] = radius_edges

    return final_bins


def build_voxelizer(final_bins, n_bins=(8, 8, 6), dim_sweep_order=(0, 1, 2), revert_order=(0, 1, 2)):
    """
    Construct a working fm4npp.datasets.voxelizer.Voxelizer instance from a
    precomputed final_bins dict, bypassing __init__'s file-loading /
    dataset-DataLoader requirement entirely. Replicates exactly what __init__
    does AFTER it obtains final_bins (whether from pickle or compute_bins) --
    tokenize()/grouping() only depend on self.final_bins and self.refarray,
    both set identically here.

    dim_sweep_order / revert_order default to Voxelizer's own class defaults
    ([0,1,2], [0,1,2]). We have no record of what order the m3/k30 checkpoint
    was actually pretrained with -- if downstream results look wrong, this is
    one of the first things worth varying (there's no way to recover the true
    value without the original training config, so treat this as a documented
    assumption, not a confirmed fact).
    """
    from fm4npp.datasets.voxelizer import Voxelizer

    v = Voxelizer.__new__(Voxelizer)  # skip __init__ entirely
    v.data_proxy = None
    v.n_bins = n_bins
    v.final_bins = final_bins
    v.all_bins = v.get_unique_voxel_IDs  # replicate original's (dead, unused) assignment
    v.dim_sweep_order = list(dim_sweep_order)
    v.revert_order = list(revert_order)
    v.init_group_stats(dim_sweep_order=list(dim_sweep_order), revert_order=list(revert_order))
    return v


def hierarchical_raster_scan_order(voxelizer, eta_n, phi_n, r_n):
    """
    Given a fitted Voxelizer and normalized (eta, phi, r) arrays for ONE
    event, return the permutation indices implementing the real Hierarchical
    Raster Scan: inter-box order (global, outward-progressing) with intra-box
    order by raw r ascending (local track continuity) -- matching the paper's
    description exactly.

    voxelizer.grouping() already returns each point's box index in the
    correct traversal order (boxes were enumerated in bins_in_order sequence
    when building refarray in init_group_stats), so sorting by
    (group_id, r) directly gives the correct hierarchical order -- no extra
    rank lookup needed.
    """
    tensor = torch.from_numpy(np.stack([eta_n, phi_n, r_n], axis=-1)).float().unsqueeze(0)
    voxelized = voxelizer.tokenize(tensor, start_idx=0)          # (N, 3) bin indices per point
    group_id = voxelizer.grouping(voxelized.unsqueeze(0)).numpy()  # (N,) box index, correct traversal order

    order = np.lexsort((r_n, group_id))  # primary: group_id ascending, secondary: r ascending
    return order