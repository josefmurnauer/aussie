import logging

import awkward as ak
import numpy as np
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
# Minimal branch subsets -- only what's needed for the 4 scalar features
# (much faster to load than the full WWbbData token representation)
# ----------------------------------------------------------------------------

RECO_BRANCHES_MULTI = {
    k: RECO_BRANCHES[k] for k in ("mu_pt", "met_met", "jet_pt", "weight")
}
PARTICLE_BRANCHES_MULTI = {
    k: PARTICLE_BRANCHES[k] for k in ("mu_pt", "met_met", "jet_pt")
}

# feature order: [leading jet pT, subleading jet pT, MET, muon pT]
FEATURE_NAMES = ["j1_pt", "j2_pt", "met_pt", "mu_pt"]
N_FEATURES = 4


def _extract_scalar_features(arrays, branch_map, n_events, drop_invalid=False):
    """Extract 4 scalar pT features: [j1_pt, j2_pt, met_pt, mu_pt].

    If drop_invalid is True, returns an additional boolean `valid` mask
    marking events where all features are finite (no NaN truth muon).
    Otherwise, NaNs are zero-filled (only safe for reco features, which
    should never contain NaN in practice).
    """
    mu_pt = _get_object_field(arrays, branch_map["mu_pt"], scale=MEV_TO_GEV)
    met_pt = _get_object_field(arrays, branch_map["met_met"], scale=MEV_TO_GEV)

    jets_pt = _pad(arrays[branch_map["jet_pt"]], 2, scale=MEV_TO_GEV)
    j1_pt = jets_pt[:, 0]
    j2_pt = jets_pt[:, 1]

    features = np.stack([j1_pt, j2_pt, met_pt, mu_pt], axis=1).astype(np.float32)

    valid = np.isfinite(features).all(axis=1)
    n_invalid = (~valid).sum()

    if n_invalid > 0:
        if drop_invalid:
            log.warning(
                f"{n_invalid} / {n_events} events have non-finite features "
                f"(likely non-muon truth lepton) -- will be dropped"
            )
        else:
            log.warning(
                f"Replacing {n_invalid} non-finite value(s) with 0.0"
            )
            features = np.nan_to_num(features, nan=0.0)

    return torch.from_numpy(features).float(), torch.from_numpy(valid)


def _needed_columns_multi(branch_map, include_weight=False):
    cols = [v for k, v in branch_map.items() if k != "weight"]
    if include_weight and "weight" in branch_map:
        cols.append(branch_map["weight"])
    return cols


def _load_arrays_multi(paths, columns, num=None, seed=42):
    """Same random-sampling logic as WWbbData's loader, kept local here
    to avoid depending on the internal helper name in wwbb.py."""
    if len(paths) == 1:
        arrays = ak.from_parquet(paths[0], columns=columns)
    else:
        arrays = ak.concatenate(
            [ak.from_parquet(p, columns=columns) for p in paths]
        )

    total = len(arrays)
    if num is not None and num < total:
        rng = np.random.default_rng(seed)
        indices = rng.choice(total, size=num, replace=False)
        indices.sort()
        arrays = arrays[indices]
        log.info(f"  randomly sampled {num} / {total} events (seed={seed})")
    elif num is not None and num >= total:
        log.info(f"  requested {num} but only {total} available -- using all")

    return arrays


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

@tensorclass
class WWbbMultiData(UnfoldingData):
    """
    Low-dimensional (4 scalar pT features) version of the WWbb dataset,
    for use with a plain MLP classifier + KernelUnfolder, as opposed to
    the full L-GATr point-cloud approach in WWbbData.

    Features (reco and truth, same ordering):
        0 : leading jet pT
        1 : subleading jet pT
        2 : MET
        3 : muon pT

    Reads from the SAME parquet files as WWbbData (produced by the ROOT
    converter) -- no separate data preparation needed.
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
            label = "sim" if i == 0 else "pseudodata"
            log.info(f"[{label}] reading {len(paths)} parquet file(s) from {path}")

            columns = _needed_columns_multi(RECO_BRANCHES_MULTI, include_weight=True)
            columns += _needed_columns_multi(PARTICLE_BRANCHES_MULTI)
            columns = list(dict.fromkeys(columns))

            arrays = _load_arrays_multi(paths, columns, num=num_i, seed=seed)
            n_events = len(arrays)
            log.info(f"[{label}] loaded {n_events} events")

            # ---------------- RECO LEVEL ----------------
            x, valid_x = _extract_scalar_features(
                arrays, RECO_BRANCHES_MULTI, n_events, drop_invalid=False
            )

            # ---------------- PARTICLE LEVEL ----------------
            # drop events with invalid truth muon (sim only -- pseudodata's z
            # is never used in training, but we filter consistently here too
            # for clean truth-vs-truth comparison plots)
            z, valid_z = _extract_scalar_features(
                arrays, PARTICLE_BRANCHES_MULTI, n_events, drop_invalid=True
            )

            keep_mask = valid_z if i == 0 else torch.ones(n_events, dtype=torch.bool)
            n_kept = keep_mask.sum().item()
            if n_kept < n_events:
                log.info(f"[{label}] dropping {n_events - n_kept} events with invalid truth muon")

            x = x[keep_mask]
            z = torch.nan_to_num(z[keep_mask], nan=0.0)  # safety net, should be moot now
            n_events = n_kept

            tensor_kwargs["x"].append(x)
            tensor_kwargs["z"].append(z)

            weight_branch = RECO_BRANCHES_MULTI["weight"]
            if weight_branch in arrays.fields:
                w = ak.to_numpy(arrays[weight_branch]).astype(np.float32)
            else:
                w = np.ones(len(arrays), dtype=np.float32)
            w = w[keep_mask.numpy()]
            raw_weights.append(w)

            tensor_kwargs["labels"].append(
                torch.full((n_events,), i, dtype=torch.float32)
            )

            batch_size += n_events

        # weights + rest unchanged from here
        if normalize:
            w_sim_raw, w_data_raw = raw_weights[0], raw_weights[1]
            sum_sim = np.abs(w_sim_raw).sum()
            sum_data = np.abs(w_data_raw).sum()
            target = batch_size / 2.0
            scale_sim = target / sum_sim
            scale_data = target / sum_data
            normalized_weights = [w_sim_raw * scale_sim, w_data_raw * scale_data]
        else:
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
    Preprocessing for the 4 scalar pT features:
        1. LogScale.forward : x -> log(x + eps)   (all 4 are pT values)
        2. ShiftAndScale.forward : standardize to ~N(0, 1)

    NOTE: shift_x/scale_x/shift_z/scale_z below are PLACEHOLDERS.
    Run compute_shift_scale() once on your sim data and paste the
    printed values into WWbbMultiProcess before training.
    """

    shift_x: torch.Tensor
    shift_z: torch.Tensor
    scale_x: torch.Tensor
    scale_z: torch.Tensor
    eps: float = 1e-3

    _log_indices: tuple = (0, 1, 2, 3)

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
    Compute per-feature mean/std after log transform from the sim parquet
    files, for both reco (x) and truth (z) levels.

    Events with invalid (non-finite) truth-level features are excluded
    from the truth-level (z) statistics, consistent with how WWbbMultiData.read()
    drops these events entirely during training.

    Usage:
        python -c "
        from src.datasets.wwbb_multi import compute_shift_scale
        compute_shift_scale('/scratch/mjosef/Unfolding/aussie/data/sim', num=2_000_000)
        "
    """
    eps = 1e-3
    log_idx = [0, 1, 2, 3]

    paths = _resolve_paths(path_sim)
    columns = _needed_columns_multi(RECO_BRANCHES_MULTI, include_weight=False)
    columns += _needed_columns_multi(PARTICLE_BRANCHES_MULTI)
    columns = list(dict.fromkeys(columns))

    arrays = _load_arrays_multi(paths, columns, num=num, seed=seed)
    n_events = len(arrays)

    x, _ = _extract_scalar_features(arrays, RECO_BRANCHES_MULTI, n_events, drop_invalid=False)
    z, valid_z = _extract_scalar_features(arrays, PARTICLE_BRANCHES_MULTI, n_events, drop_invalid=True)

    n_dropped = (~valid_z).sum().item()
    if n_dropped > 0:
        print(f"Dropping {n_dropped} / {n_events} events with invalid truth muon "
              f"before computing z-level statistics")

    z = z[valid_z]

    x = LogScale.forward(x.clone(), indices=log_idx, eps=eps)
    z = LogScale.forward(z.clone(), indices=log_idx, eps=eps)

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
            shift_x=torch.tensor([4.809, 4.4378, 4.0433, 4.2637]),
            scale_x=torch.tensor([0.468, 0.4018, 0.7365, 0.4099]),
            shift_z=torch.tensor([4.837, 4.4723, 3.8867, 4.2691]),
            scale_z=torch.tensor([0.4589, 0.4074, 0.7764, 0.4098]),
                 ),
    )

    observables_x: Tuple[Observable] = (
        Observable(
            name="j1_pt", compute=lambda x: x[..., 0],
            label=r"Leading jet $p_T$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
        ),
        Observable(
            name="j2_pt", compute=lambda x: x[..., 1],
            label=r"Subleading jet $p_T$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
        ),
        Observable(
            name="met_pt", compute=lambda x: x[..., 2],
            label=r"$E_T^{\rm miss}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
        ),
        Observable(
            name="mu_pt", compute=lambda x: x[..., 3],
            label=r"$p_{T,\mu}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
        ),
    )

    observables_z: Tuple[Observable] = (
        Observable(
            name="j1_pt_truth", compute=lambda z: z[..., 0],
            label=r"Leading jet $p_T^{\rm truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
        ),
        Observable(
            name="j2_pt_truth", compute=lambda z: z[..., 1],
            label=r"Subleading jet $p_T^{\rm truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
        ),
        Observable(
            name="met_pt_truth", compute=lambda z: z[..., 2],
            label=r"$E_T^{\rm miss,truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
        ),
        Observable(
            name="mu_pt_truth", compute=lambda z: z[..., 3],
            label=r"$p_{T,\mu}^{\rm truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
        ),
    )