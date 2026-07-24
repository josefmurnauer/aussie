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
# Branch subsets. Only muon and jet pt/energy are needed for this feature set.
# ----------------------------------------------------------------------------

RECO_BRANCHES_MULTI = {
    k: RECO_BRANCHES[k]
    for k in ("mu_pt", "mu_e", "jet_pt", "jet_e", "weight")
}

PARTICLE_BRANCHES_MULTI = {
    k: PARTICLE_BRANCHES[k]
    for k in ("mu_pt", "mu_e", "jet_pt", "jet_e")
}

N_LEADING_JETS = 3

# feature order: [mu_pt, mu_e, j1_pt, j1_e, j2_pt, j2_e, j3_pt, j3_e]
FEATURE_NAMES = [
    "mu_pt", "mu_e",
    "j1_pt", "j1_e",
    "j2_pt", "j2_e",
    "j3_pt", "j3_e",
]
N_FEATURES = 2 + 2 * N_LEADING_JETS  # 8

# all features are strictly-positive pT/energy quantities -- log-transform all
LOG_INDICES = tuple(range(N_FEATURES))


def _available_columns(paths):
    """Return the set of column names available in the parquet file(s) for
    a given population, by inspecting the schema of the first file. Used
    to detect whether particle-level truth branches and/or the MC weight
    branch exist for a given population -- e.g. real collision data will
    have neither, while MC sim/pseudodata will have both."""
    return set(pq.ParquetFile(paths[0]).schema_arrow.names)


def _extract_scalar_features(arrays, branch_map, n_events, drop_invalid=False):
    """Extract 8 scalar features:
        [mu_pt, mu_e, j1_pt, j1_e, j2_pt, j2_e, j3_pt, j3_e]

    Jets are assumed pT-sorted (standard ATLAS ntuple convention).
    Events with fewer than N_LEADING_JETS jets get 0 for the missing
    jet slots' pt/e.
    """
    mu_pt = _get_object_field(arrays, branch_map["mu_pt"], scale=MEV_TO_GEV)
    mu_e = _get_object_field(arrays, branch_map["mu_e"], scale=MEV_TO_GEV)

    jets_pt = _pad(arrays[branch_map["jet_pt"]], N_LEADING_JETS, scale=MEV_TO_GEV)
    jets_e = _pad(arrays[branch_map["jet_e"]], N_LEADING_JETS, scale=MEV_TO_GEV)

    features = np.stack(
        [
            mu_pt, mu_e,
            jets_pt[:, 0], jets_e[:, 0],
            jets_pt[:, 1], jets_e[:, 1],
            jets_pt[:, 2], jets_e[:, 2],
        ],
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
        0 : muon pT
        1 : muon E
        2 : leading jet pT
        3 : leading jet E
        4 : subleading jet pT
        5 : subleading jet E
        6 : 3rd jet pT
        7 : 3rd jet E

    path_data may point at either Herwig pseudodata (MC, has particle-level
    truth and weight_total_NOSYS) or real collision data (has neither).
    Availability is detected automatically per population by inspecting
    the parquet schema.
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
            log.info(f"[{label}] has_weight={has_weight}, has_truth={has_truth}")

            columns = _needed_columns_multi(RECO_BRANCHES_MULTI, include_weight=has_weight)
            if has_truth:
                columns += _needed_columns_multi(PARTICLE_BRANCHES_MULTI)
            columns = list(dict.fromkeys(columns))

            arrays, weight_scale = _load_arrays_multi(paths, columns, num=num_i, seed=seed)
            n_events = len(arrays)
            log.info(f"[{label}] loaded {n_events} events")

            # ---------------- RECO LEVEL ----------------
            x, _ = _extract_scalar_features(
                arrays, RECO_BRANCHES_MULTI, n_events, drop_invalid=False
            )

            # ---------------- PARTICLE LEVEL ----------------
            if has_truth:
                z, valid_z = _extract_scalar_features(
                    arrays, PARTICLE_BRANCHES_MULTI, n_events, drop_invalid=True
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

            # event weights, compensated for subsampling so relative
            # normalization between populations is preserved regardless of
            # num_sim/num_data. Defaults to unit weight if the weight
            # branch isn't present (always true for real collision data).
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
    Preprocessing for the 8 scalar features (all pT/energy quantities):
        log-transform + standardize, applied uniformly to all 8 indices.

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
    Compute per-feature mean/std after log transform from the sim parquet
    files, for both reco (x) and truth (z) levels.

    Usage:
        python -c "
        from src.datasets.wwbb_multi import compute_shift_scale
        compute_shift_scale('/scratch/mjosef/Unfolding/aussie/data/sim', num=2_000_000)
        "
    """
    eps = 1e-3

    paths = _resolve_paths(path_sim)
    columns = _needed_columns_multi(RECO_BRANCHES_MULTI, include_weight=False)
    columns += _needed_columns_multi(PARTICLE_BRANCHES_MULTI)
    columns = list(dict.fromkeys(columns))

    arrays, _ = _load_arrays_multi(paths, columns, num=num, seed=seed)
    n_events = len(arrays)

    x, _ = _extract_scalar_features(arrays, RECO_BRANCHES_MULTI, n_events, drop_invalid=False)
    z, valid_z = _extract_scalar_features(arrays, PARTICLE_BRANCHES_MULTI, n_events, drop_invalid=True)

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
            shift_x=torch.tensor([4.2637, 4.7416, 4.809, 5.4326, 4.4378, 5.106, 4.1289, 4.8796]),
            scale_x=torch.tensor([0.4099, 0.5975, 0.468, 0.7693, 0.4018, 0.7723, 0.3284, 0.82]),
            shift_z=torch.tensor([4.2691, 4.7469, 4.837, 5.4406, 4.4723, 5.1132, 4.1261, 4.8485]),
            scale_z=torch.tensor([0.4098, 0.5971, 0.4589, 0.7488, 0.4074, 0.7497, 0.3686, 0.8001]),
        ),
    )

    observables_x: Tuple[Observable] = (
        Observable(
            name="mu_pt", compute=lambda x: x[..., 0],
            label=r"$p_{T,\mu}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="mu_e", compute=lambda x: x[..., 1],
            label=r"$E_{\mu}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="j1_pt", compute=lambda x: x[..., 2],
            label=r"Leading jet $p_T$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="j1_e", compute=lambda x: x[..., 3],
            label=r"Leading jet $E$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="j2_pt", compute=lambda x: x[..., 4],
            label=r"Subleading jet $p_T$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="j2_e", compute=lambda x: x[..., 5],
            label=r"Subleading jet $E$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="j3_pt", compute=lambda x: x[..., 6],
            label=r"3rd jet $p_T$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="j3_e", compute=lambda x: x[..., 7],
            label=r"3rd jet $E$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
    )

    observables_z: Tuple[Observable] = (
        Observable(
            name="mu_pt_truth", compute=lambda z: z[..., 0],
            label=r"$p_{T,\mu}^{\rm truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="mu_e_truth", compute=lambda z: z[..., 1],
            label=r"$E_{\mu}^{\rm truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="j1_pt_truth", compute=lambda z: z[..., 2],
            label=r"Leading jet $p_T^{\rm truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="j1_e_truth", compute=lambda z: z[..., 3],
            label=r"Leading jet $E^{\rm truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="j2_pt_truth", compute=lambda z: z[..., 4],
            label=r"Subleading jet $p_T^{\rm truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="j2_e_truth", compute=lambda z: z[..., 5],
            label=r"Subleading jet $E^{\rm truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="j3_pt_truth", compute=lambda z: z[..., 6],
            label=r"3rd jet $p_T^{\rm truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
        Observable(
            name="j3_e_truth", compute=lambda z: z[..., 7],
            label=r"3rd jet $E^{\rm truth}$ [GeV]", qlims=(1e-3, 1 - 1e-3), logy=True,
            log_bins=True,
        ),
    )