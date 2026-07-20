import logging

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import torch

from collections import defaultdict
from dataclasses import dataclass
from tensordict import tensorclass
from typing import Callable, List, Tuple, Optional, Union

from src.datasets.base_dataset import UnfoldingData
from src.utils.observable import Observable
from src.utils.transforms import ShiftAndScale, LogScale
from src.datasets.wwbb import (
    RECO_BRANCHES,
    PARTICLE_BRANCHES,
    MEV_TO_GEV,
    _resolve_paths,
    _get_object_field,
    _pad,
)

log = logging.getLogger("WWbbMultiData")


# ----------------------------------------------------------------------------
# Branch subsets. RECO_BRANCHES/PARTICLE_BRANCHES (from wwbb.py) don't
# include the dedicated single leading-b-jet branches, so those are added
# directly here.
# ----------------------------------------------------------------------------

RECO_BRANCHES_MULTI = {
    k: RECO_BRANCHES[k]
    for k in ("mu_pt", "mu_phi", "met_met", "jet_pt", "jet_phi", "weight")
}
RECO_BRANCHES_MULTI["bjet_pt"] = "jet1_bjet_pt_NOSYS"   # already in GeV

PARTICLE_BRANCHES_MULTI = {
    k: PARTICLE_BRANCHES[k]
    for k in ("mu_pt", "mu_phi", "met_met", "jet_pt", "jet_phi")
}
PARTICLE_BRANCHES_MULTI["bjet_pt"] = "PL_bjet1_pt_GEV_NOSYS"   # already in GeV

# feature order:
# [j1_pt, j2_pt, met_pt, mu_pt, n_jets, dphi(mu,j1), dphi(mu,j2), bjet_pt]
FEATURE_NAMES = [
    "j1_pt", "j2_pt", "met_pt", "mu_pt",
    "n_jets", "dphi_mu_j1", "dphi_mu_j2", "bjet_pt",
]
N_FEATURES = 8

# indices that get log-transformed (strictly positive, wide dynamic range
# pT-like quantities). n_jets and the two delta_phi features are NOT
# log-transformed -- they're standardized directly.
LOG_INDICES = (0, 1, 2, 3, 7)


def _delta_phi(phi1, phi2):
    """Wrap to [-pi, pi]."""
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def _available_columns(paths):
    """Return the set of column names available in the parquet file(s) for
    a given population, by inspecting the schema of the first file. Used
    to detect whether particle-level truth branches and/or the MC weight
    branch exist for a given population -- e.g. real collision data will
    have neither, while MC sim/pseudodata will have both."""
    return set(pq.ParquetFile(paths[0]).schema_arrow.names)


def _extract_scalar_features(arrays, branch_map, n_events, drop_invalid=False,
                              has_bjet=True):
    """Extract scalar features:
        [j1_pt, j2_pt, met_pt, mu_pt, n_jets,
         delta_phi(mu, j1), delta_phi(mu, j2), (bjet_pt)]

    Jets are assumed pT-sorted (standard ATLAS ntuple convention).
    Events with fewer than 2 jets get 0 for the missing subleading-jet
    slot and the corresponding delta_phi (n_jets correctly reflects this
    so the network can learn these features are meaningless in that case).

    bjet_pt comes from the ntuple's precomputed leading-b-jet branch
    (already in GeV). If has_bjet is False (branch not present in this
    population's schema, e.g. an alternate real-data ntuple production
    without the dedicated single-b-jet branch), bjet_pt is filled with 0
    for every event rather than reading a nonexistent column.
    """
    mu_pt = _get_object_field(arrays, branch_map["mu_pt"], scale=MEV_TO_GEV)
    mu_phi = _get_object_field(arrays, branch_map["mu_phi"])
    met_pt = _get_object_field(arrays, branch_map["met_met"], scale=MEV_TO_GEV)

    n_jets = ak.num(arrays[branch_map["jet_pt"]]).to_numpy().astype(np.float32)

    jets_pt = _pad(arrays[branch_map["jet_pt"]], 2, scale=MEV_TO_GEV)
    jets_phi = _pad(arrays[branch_map["jet_phi"]], 2)
    j1_pt = jets_pt[:, 0]
    j2_pt = jets_pt[:, 1]
    j1_phi = jets_phi[:, 0]
    j2_phi = jets_phi[:, 1]

    dphi_mu_j1 = _delta_phi(mu_phi, j1_phi).astype(np.float32)
    dphi_mu_j2 = _delta_phi(mu_phi, j2_phi).astype(np.float32)
    # zero out delta_phi where the corresponding jet doesn't exist
    dphi_mu_j1 = np.where(n_jets > 0, dphi_mu_j1, 0.0)
    dphi_mu_j2 = np.where(n_jets > 1, dphi_mu_j2, 0.0)

    if has_bjet:
        # leading b-jet: already in GeV, no MEV_TO_GEV scaling
        bjet_pt = _get_object_field(arrays, branch_map["bjet_pt"], scale=1.0)
        n_no_bjet = (bjet_pt <= 0).sum()
        if n_no_bjet > 0:
            log.info(
                f"  {n_no_bjet} / {n_events} events have no identified "
                f"leading b-jet (bjet_pt <= 0) -- clipped to 0"
            )
        bjet_pt = np.clip(bjet_pt, a_min=0.0, a_max=None)
    else:
        log.info(
            f"  bjet_pt branch not present in this population's schema "
            f"-- filling with 0 for all {n_events} events"
        )
        bjet_pt = np.zeros(n_events, dtype=np.float32)

    features = np.stack(
        [j1_pt, j2_pt, met_pt, mu_pt, n_jets, dphi_mu_j1, dphi_mu_j2, bjet_pt],
        axis=1,
    ).astype(np.float32)

    valid = np.isfinite(features).all(axis=1)
    n_invalid = (~valid).sum()

    if n_invalid > 0:
        if drop_invalid:
            log.warning(
                f"{n_invalid} / {n_events} events have non-finite features "
                f"(likely non-muon truth lepton) -- will be dropped"
            )
        else:
            log.warning(f"Replacing {n_invalid} non-finite value(s) with 0.0")
            features = np.nan_to_num(features, nan=0.0)

    return torch.from_numpy(features).float(), torch.from_numpy(valid)


def _needed_columns_multi(branch_map, include_weight=False):
    cols = [v for k, v in branch_map.items() if k != "weight"]
    if include_weight and "weight" in branch_map:
        cols.append(branch_map["weight"])
    return cols


def _load_arrays_multi(paths, columns, num=None, seed=42):
    """Load one or more parquet files as a single Awkward Array,
    optionally drawing a random subset of `num` events uniformly across
    ALL files. Returns a compensation factor (total/num) so that the sum
    of weights in the subsample remains an unbiased estimator of the
    full-sample weight sum -- see the identical helper in wwbb.py for the
    full rationale."""
    if len(paths) == 1:
        arrays = ak.from_parquet(paths[0], columns=columns)
    else:
        arrays = ak.concatenate(
            [ak.from_parquet(p, columns=columns) for p in paths]
        )

    total = len(arrays)
    weight_scale = 1.0

    if num is not None and num < total:
        rng = np.random.default_rng(seed)
        indices = rng.choice(total, size=num, replace=False)
        indices.sort()
        arrays = arrays[indices]
        weight_scale = total / num
        log.info(
            f"  randomly sampled {num} / {total} events "
            f"(seed={seed}, weight compensation factor={weight_scale:.4f})"
        )
    elif num is not None and num >= total:
        log.info(f"  requested {num} but only {total} available -- using all")

    return arrays, weight_scale


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

@tensorclass
class WWbbMultiData(UnfoldingData):
    """
    Low-dimensional scalar-feature version of the WWbb dataset, for use
    with a plain MLP classifier + KernelUnfolder/AutoDiffUnfolder, as
    opposed to the full L-GATr point-cloud approach in WWbbData.

    Features (reco and truth, same ordering):
        0 : leading jet pT
        1 : subleading jet pT
        2 : MET
        3 : muon pT
        4 : number of jets
        5 : delta_phi(muon, leading jet)
        6 : delta_phi(muon, subleading jet)
        7 : leading b-jet pT

    path_data may point at either Herwig pseudodata (MC, has particle-level
    truth, weight_total_NOSYS, and the leading-b-jet branch) or real
    collision data (has none of these). Availability is detected
    automatically per population by inspecting the parquet schema.
    """

    @classmethod
    def read(
        cls,
        path_sim: Union[str, List[str]],
        path_data: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_sim: Optional[int] = None,
        num_data: Optional[int] = None,
        num: Optional[int] = None,
        normalize: bool = False,
        seed: int = 42,
    ):
        if num is not None:
            if num_sim is None:
                num_sim = num
            if num_data is None:
                num_data = num

        batch_size = 0
        tensor_kwargs = defaultdict(list)
        raw_weights = []

        for i, (path, num_i) in enumerate(
            ((path_sim, num_sim), (path_data, num_data))
        ):
            paths = _resolve_paths(path)
            label = "sim" if i == 0 else "data"
            log.info(f"[{label}] reading {len(paths)} parquet file(s) from {path}")

            available = _available_columns(paths)
            has_weight = RECO_BRANCHES_MULTI["weight"] in available
            has_truth = all(b in available for b in PARTICLE_BRANCHES_MULTI.values())
            has_bjet_reco = RECO_BRANCHES_MULTI["bjet_pt"] in available
            has_bjet_truth = PARTICLE_BRANCHES_MULTI["bjet_pt"] in available
            log.info(
                f"[{label}] has_weight={has_weight}, has_truth={has_truth}, "
                f"has_bjet_reco={has_bjet_reco}, has_bjet_truth={has_bjet_truth}"
            )

            columns = _needed_columns_multi(RECO_BRANCHES_MULTI, include_weight=has_weight)
            if not has_bjet_reco:
                columns = [c for c in columns if c != RECO_BRANCHES_MULTI["bjet_pt"]]
            if has_truth:
                truth_cols = _needed_columns_multi(PARTICLE_BRANCHES_MULTI)
                if not has_bjet_truth:
                    truth_cols = [
                        c for c in truth_cols if c != PARTICLE_BRANCHES_MULTI["bjet_pt"]
                    ]
                columns += truth_cols
            columns = list(dict.fromkeys(columns))

            arrays, weight_scale = _load_arrays_multi(paths, columns, num=num_i, seed=seed)
            n_events = len(arrays)
            log.info(f"[{label}] loaded {n_events} events")

            # ---------------- RECO LEVEL ----------------
            x, _ = _extract_scalar_features(
                arrays, RECO_BRANCHES_MULTI, n_events,
                drop_invalid=False, has_bjet=has_bjet_reco,
            )

            # ---------------- PARTICLE LEVEL ----------------
            if has_truth:
                z, valid_z = _extract_scalar_features(
                    arrays, PARTICLE_BRANCHES_MULTI, n_events,
                    drop_invalid=True, has_bjet=has_bjet_truth,
                )
                keep_mask = valid_z
            else:
                # placeholder zeros -- never used in training for this
                # population, but kept shape-consistent so batching works
                z = torch.zeros((n_events, N_FEATURES))
                keep_mask = torch.ones(n_events, dtype=torch.bool)

            n_kept = keep_mask.sum().item()
            if n_kept < n_events:
                log.info(
                    f"[{label}] dropping {n_events - n_kept} events "
                    f"with invalid truth muon"
                )

            x = x[keep_mask]
            z = torch.nan_to_num(z[keep_mask], nan=0.0)
            n_events = n_kept

            tensor_kwargs["x"].append(x)
            tensor_kwargs["z"].append(z)

            # event weights, compensated for subsampling. Defaults to unit
            # weight if the weight branch isn't present (always true for
            # real collision data).
            if has_weight:
                w = ak.to_numpy(arrays[RECO_BRANCHES_MULTI["weight"]]).astype(np.float32)
            else:
                w = np.ones(len(arrays), dtype=np.float32)
            w = w[keep_mask.numpy()]
            w = w * weight_scale
            raw_weights.append(w)

            tensor_kwargs["labels"].append(
                torch.full((n_events,), i, dtype=torch.float32)
            )

            batch_size += n_events

        if normalize:
            w_sim_raw, w_data_raw = raw_weights[0], raw_weights[1]
            sum_sim = np.abs(w_sim_raw).sum()
            sum_data = np.abs(w_data_raw).sum()
            target = batch_size / 2.0
            scale_sim = target / sum_sim
            scale_data = target / sum_data
            log.info(
                f"Normalizing weights: sim scale={scale_sim:.4f}, "
                f"data scale={scale_data:.4f}"
            )
            normalized_weights = [w_sim_raw * scale_sim, w_data_raw * scale_data]
        else:
            log.info("Skipping weight normalization (normalize=False)")
            normalized_weights = raw_weights

        for w in normalized_weights:
            logw = np.log(np.abs(w) + 1e-12)
            tensor_kwargs["sample_logweights"].append(torch.from_numpy(logw).float())

        for k in tensor_kwargs:
            tensor_kwargs[k] = torch.cat(tensor_kwargs[k], dim=0)

        return cls(batch_size=[batch_size], device=device, **tensor_kwargs)


# ----------------------------------------------------------------------------
# Transform
# ----------------------------------------------------------------------------

@tensorclass
class WWbbMultiTransform:
    """
    Preprocessing for the 8 scalar features:
        indices 0,1,2,3,7 (pT features): log-transform + standardize
        index 4 (n_jets): standardize only
        indices 5,6 (delta_phi): standardize only

    NOTE: shift_x/scale_x/shift_z/scale_z below are PLACEHOLDERS.
    Run compute_shift_scale() once on your sim data and paste the
    printed values into WWbbMultiProcess before training.
    """

    shift_x: torch.Tensor
    shift_z: torch.Tensor
    scale_x: torch.Tensor
    scale_z: torch.Tensor
    eps: float = 1e-3

    _log_indices: tuple = LOG_INDICES

    def forward(self, batch):
        batch.x = LogScale.forward(batch.x, indices=list(self._log_indices), eps=self.eps)
        batch.z = LogScale.forward(batch.z, indices=list(self._log_indices), eps=self.eps)

        batch.x = ShiftAndScale.forward(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.z = ShiftAndScale.forward(batch.z, shift=self.shift_z, scale=self.scale_z)

        return batch

    def reverse(self, batch):
        batch.x = ShiftAndScale.reverse(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.z = ShiftAndScale.reverse(batch.z, shift=self.shift_z, scale=self.scale_z)

        batch.x = LogScale.reverse(batch.x, indices=list(self._log_indices))
        batch.z = LogScale.reverse(batch.z, indices=list(self._log_indices))

        return batch


# ----------------------------------------------------------------------------
# Utility: compute shift/scale from your sim data
# ----------------------------------------------------------------------------

def compute_shift_scale(path_sim: str, num: Optional[int] = None, seed: int = 42):
    """
    Compute per-feature mean/std after log transform (for pT-like features
    only) from the sim parquet files, for both reco (x) and truth (z)
    levels.

    Usage:
        python -c "
        from src.datasets.wwbb_multi import compute_shift_scale
        compute_shift_scale('/scratch/mjosef/Unfolding/aussie/data/sim', num=2_000_000)
        "
    """
    eps = 1e-3

    paths = _resolve_paths(path_sim)
    available = _available_columns(paths)
    has_bjet_reco = RECO_BRANCHES_MULTI["bjet_pt"] in available
    has_bjet_truth = PARTICLE_BRANCHES_MULTI["bjet_pt"] in available

    columns = _needed_columns_multi(RECO_BRANCHES_MULTI, include_weight=False)
    if not has_bjet_reco:
        columns = [c for c in columns if c != RECO_BRANCHES_MULTI["bjet_pt"]]
    truth_cols = _needed_columns_multi(PARTICLE_BRANCHES_MULTI)
    if not has_bjet_truth:
        truth_cols = [c for c in truth_cols if c != PARTICLE_BRANCHES_MULTI["bjet_pt"]]
    columns += truth_cols
    columns = list(dict.fromkeys(columns))

    arrays, _ = _load_arrays_multi(paths, columns, num=num, seed=seed)
    n_events = len(arrays)

    x, _ = _extract_scalar_features(
        arrays, RECO_BRANCHES_MULTI, n_events, drop_invalid=False, has_bjet=has_bjet_reco
    )
    z, valid_z = _extract_scalar_features(
        arrays, PARTICLE_BRANCHES_MULTI, n_events, drop_invalid=True, has_bjet=has_bjet_truth
    )

    n_dropped = (~valid_z).sum().item()
    if n_dropped > 0:
        print(f"Dropping {n_dropped} / {n_events} events with invalid truth muon "
              f"before computing z-level statistics")

    z = z[valid_z]

    x = LogScale.forward(x.clone(), indices=list(LOG_INDICES), eps=eps)
    z = LogScale.forward(z.clone(), indices=list(LOG_INDICES), eps=eps)

    shift_x, scale_x = x.mean(0), x.std(0)
    shift_z, scale_z = z.mean(0), z.std(0)

    print("Paste into WWbbMultiProcess:\n")
    print(f"  shift_x=torch.tensor({[round(v, 4) for v in shift_x.tolist()]}),")
    print(f"  scale_x=torch.tensor({[round(v, 4) for v in scale_x.tolist()]}),")
    print(f"  shift_z=torch.tensor({[round(v, 4) for v in shift_z.tolist()]}),")
    print(f"  scale_z=torch.tensor({[round(v, 4) for v in scale_z.tolist()]}),")

    return shift_x, scale_x, shift_z, scale_z


# ----------------------------------------------------------------------------
# Process descriptor
# ----------------------------------------------------------------------------

@dataclass
class WWbbMultiProcess:
    num_features: int = N_FEATURES

    transforms: Tuple[Callable] = (
        WWbbMultiTransform(
            shift_x=torch.tensor([4.8091, 4.4381, 4.0437, 4.2637, 5.8316, -0.0007, -0.0, 4.4389]),
            scale_x=torch.tensor([0.4682, 0.402, 0.7365, 0.41, 2.2003, 2.2351, 1.9978, 0.5334]),
            shift_z=torch.tensor([4.837, 4.4725, 3.8872, 4.2691, 5.1193, -0.0006, 0.0009, 4.4478]),
            scale_z=torch.tensor([0.4591, 0.4076, 0.7767, 0.4099, 1.4871, 2.2331, 1.9842, 1.3517]),
        ),
    )

    observables_x: Tuple[Observable] = (
        Observable(
            name="j1_pt", compute=lambda x: x[..., 0],
            label=r"Leading jet $p_T$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="j2_pt", compute=lambda x: x[..., 1],
            label=r"Subleading jet $p_T$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="met_pt", compute=lambda x: x[..., 2],
            label=r"$E_T^{\rm miss}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="mu_pt", compute=lambda x: x[..., 3],
            label=r"$p_{T,\mu}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="n_jets", compute=lambda x: x[..., 4],
            label=r"$N_{\rm jets}$", discrete=1, xlims=(0, 12),
            log_bins=False,
        ),
        Observable(
            name="dphi_mu_j1", compute=lambda x: x[..., 5],
            label=r"$\Delta\phi(\mu, j_1)$", xlims=(-np.pi, np.pi),
            log_bins=False,
        ),
        Observable(
            name="dphi_mu_j2", compute=lambda x: x[..., 6],
            label=r"$\Delta\phi(\mu, j_2)$", xlims=(-np.pi, np.pi),
            log_bins=False,
        ),
        Observable(
            name="bjet_pt", compute=lambda x: x[..., 7],
            label=r"Leading $b$-jet $p_T$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
    )

    observables_z: Tuple[Observable] = (
        Observable(
            name="j1_pt_truth", compute=lambda z: z[..., 0],
            label=r"Leading jet $p_T^{\rm truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="j2_pt_truth", compute=lambda z: z[..., 1],
            label=r"Subleading jet $p_T^{\rm truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="met_pt_truth", compute=lambda z: z[..., 2],
            label=r"$E_T^{\rm miss,truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="mu_pt_truth", compute=lambda z: z[..., 3],
            label=r"$p_{T,\mu}^{\rm truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="n_jets_truth", compute=lambda z: z[..., 4],
            label=r"$N_{\rm jets}^{\rm truth}$", discrete=1, xlims=(0, 12),
            log_bins=False,
        ),
        Observable(
            name="dphi_mu_j1_truth", compute=lambda z: z[..., 5],
            label=r"$\Delta\phi(\mu, j_1)^{\rm truth}$", xlims=(-np.pi, np.pi),
            log_bins=False,
        ),
        Observable(
            name="dphi_mu_j2_truth", compute=lambda z: z[..., 6],
            label=r"$\Delta\phi(\mu, j_2)^{\rm truth}$", xlims=(-np.pi, np.pi),
            log_bins=False,
        ),
        Observable(
            name="bjet_pt_truth", compute=lambda z: z[..., 7],
            label=r"Leading $b$-jet $p_T^{\rm truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
    )