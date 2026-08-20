import glob
import logging
import os

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
from src.utils.transforms import LorentzTransform

log = logging.getLogger("WWbbData")


# ----------------------------------------------------------------------------
# Branch name mapping
# ----------------------------------------------------------------------------

RECO_BRANCHES = {
    "mu_pt":  "mu_pt_NOSYS",
    "mu_eta": "mu_eta",
    "mu_phi": "mu_phi",
    "mu_e":   "mu_e_NOSYS",

    "met_met": "met_met_NOSYS",
    "met_phi": "met_phi_NOSYS",

    "jet_pt":  "jet_pt_NOSYS",
    "jet_eta": "jet_eta",
    "jet_phi": "jet_phi",
    "jet_e":   "jet_e_NOSYS",

    "jet_gn2_pb":   "jet_GN2v01_pb",
    "jet_gn2_pc":   "jet_GN2v01_pc",
    "jet_gn2_pu":   "jet_GN2v01_pu",
    "jet_gn2_ptau": "jet_GN2v01_ptau",

    "weight": "weight_total_NOSYS",
}

PARTICLE_BRANCHES = {
    "mu_pt":  "particleLevel_PL_mu_pt",
    "mu_eta": "particleLevel_PL_mu_eta",
    "mu_phi": "particleLevel_PL_mu_phi",
    "mu_e":   "particleLevel_PL_mu_e",

    "met_met": "particleLevel_PL_met_met",
    "met_phi": "particleLevel_PL_met_phi",

    "jet_pt":  "particleLevel_PL_jet_pt",
    "jet_eta": "particleLevel_PL_jet_eta",
    "jet_phi": "particleLevel_PL_jet_phi",
    "jet_e":   "particleLevel_PL_jet_e",
}

N_JET_TAGGING_SCORES = 4  # pb, pc, pu, ptau

MEV_TO_GEV = 1e-3


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _resolve_paths(path):
    """Accept a single parquet file, a directory of parquet chunks, or a
    list/tuple mixing either, and return a flat sorted list of file paths.
    """
    if isinstance(path, (list, tuple)):
        resolved = []
        for p in path:
            resolved.extend(_resolve_paths(p))
        return resolved
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet files found in directory {path}")
        return files
    return [path]


def _available_columns(paths):
    """Return the set of column names available in the parquet file(s) for
    a given population, by inspecting the schema of the first file."""
    return set(pq.ParquetFile(paths[0]).schema_arrow.names)


def _get_object_field(arrays, field, scale=1.0):
    """Extract a per-event value for a *single-object* branch (muon, MET).

    Handles the inconsistent ATLAS ntuple convention where some single-object
    branches round-trip through Parquet as length-1 nested lists while
    others are plain per-event scalars. Rather than relying on `.ndim` alone,
    we unconditionally squeeze any leftover extra dimension after converting
    to NumPy.
    """
    x = arrays[field]
    if x.ndim > 1:
        x = ak.firsts(x)
    arr = ak.to_numpy(x).astype(np.float32)

    if arr.ndim > 1:
        assert arr.shape[1] == 1, (
            f"Expected single-object branch '{field}' to squeeze to 1D, "
            f"but got shape {arr.shape}. This branch may actually contain "
            f"more than one value per event -- check the source ntuple."
        )
        arr = arr.reshape(arr.shape[0])

    return arr * scale


def _pad(arr, max_len, scale=1.0):
    padded = ak.to_numpy(ak.fill_none(ak.pad_none(arr, max_len, clip=True), 0.0))
    return padded.astype(np.float32) * scale


def _to_cartesian(pt, eta, phi, e):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return np.stack([e, px, py, pz], axis=-1)


def _met_to_cartesian(met, met_phi):
    px = met * np.cos(met_phi)
    py = met * np.sin(met_phi)
    return np.stack([met, px, py, np.zeros_like(met)], axis=-1)


def _build_tokens(arrays, branch_map, max_jets, n_events, include_tagging=False, drop_invalid=False):
    """Build (features, mask, valid) for one event collection at one
    level (reco/truth).

    Token layout: [muon, met, jet_0, ..., jet_{max_jets-1}]
    Feature layout: [E, px, py, pz, is_mu, is_met, is_jet]

    Only the one-hot token-TYPE flags are kept as scalar auxiliary
    channels. No pT scalar and no GN2 tagging scores are included --
    `include_tagging` is accepted for call-site backward compatibility
    but is IGNORED.

    All pt/energy quantities are converted from MeV (raw ntuple units)
    to GeV.

    Some branches (most notably particle-level muon kinematics) can
    contain NaN for a small fraction of events -- e.g. when the
    truth-level lepton isn't actually a muon. `valid` marks events where
    ALL features are finite.
    """
    n_tokens = 2 + max_jets
    n_scalars = 3  # is_mu, is_met, is_jet

    mu_pt  = _get_object_field(arrays, branch_map["mu_pt"], scale=MEV_TO_GEV)
    mu_eta = _get_object_field(arrays, branch_map["mu_eta"])
    mu_phi = _get_object_field(arrays, branch_map["mu_phi"])
    mu_e   = _get_object_field(arrays, branch_map["mu_e"], scale=MEV_TO_GEV)
    mu_4vec = _to_cartesian(mu_pt, mu_eta, mu_phi, mu_e)
    assert mu_4vec.shape == (n_events, 4), f"mu_4vec shape mismatch: {mu_4vec.shape}"

    met     = _get_object_field(arrays, branch_map["met_met"], scale=MEV_TO_GEV)
    met_phi = _get_object_field(arrays, branch_map["met_phi"])
    met_4vec = _met_to_cartesian(met, met_phi)
    assert met_4vec.shape == (n_events, 4), f"met_4vec shape mismatch: {met_4vec.shape}"

    n_jets = ak.num(arrays[branch_map["jet_pt"]]).to_numpy()
    jets_4vec = _to_cartesian(
        _pad(arrays[branch_map["jet_pt"]], max_jets, scale=MEV_TO_GEV),
        _pad(arrays[branch_map["jet_eta"]], max_jets),
        _pad(arrays[branch_map["jet_phi"]], max_jets),
        _pad(arrays[branch_map["jet_e"]], max_jets, scale=MEV_TO_GEV),
    )
    assert jets_4vec.shape == (n_events, max_jets, 4), (
        f"jets_4vec shape mismatch: {jets_4vec.shape}"
    )

    momenta = np.zeros((n_events, n_tokens, 4), dtype=np.float32)
    momenta[:, 0] = mu_4vec
    momenta[:, 1] = met_4vec
    momenta[:, 2:] = jets_4vec

    # scalar layout: [is_mu, is_met, is_jet]
    scalars = np.zeros((n_events, n_tokens, n_scalars), dtype=np.float32)
    scalars[:, 0, 0] = 1.0
    scalars[:, 1, 1] = 1.0
    scalars[:, 2:, 2] = 1.0

    features = np.concatenate([momenta, scalars], axis=-1)

    # ---- NaN handling ----
    valid = np.isfinite(features).reshape(n_events, -1).all(axis=1)
    n_invalid = (~valid).sum()

    if n_invalid > 0:
        if drop_invalid:
            log.warning(
                f"{n_invalid} / {n_events} events have non-finite features "
                f"(likely non-muon truth lepton) -- will be dropped"
            )
        else:
            log.warning(f"Replacing {n_invalid} non-finite event(s) with 0.0")
            features = np.nan_to_num(features, nan=0.0)

    mask = np.zeros((n_events, n_tokens), dtype=bool)
    mask[:, :2] = True
    mask[:, 2:] = np.arange(max_jets)[None, :] < n_jets[:, None]

    return (
        torch.from_numpy(features).float(),
        torch.from_numpy(mask),
        torch.from_numpy(valid),
    )


def _needed_columns(branch_map, include_weight=False):
    cols = [v for k, v in branch_map.items() if k != "weight"]
    if include_weight and "weight" in branch_map:
        cols.append(branch_map["weight"])
    return cols


def _load_arrays(paths, columns, num=None, seed=42):
    """Load one or more parquet files as a single Awkward Array,
    optionally drawing a random subset of `num` events uniformly
    across ALL files rather than taking the first `num` events.

    Returns
    -------
    arrays : ak.Array
    weight_scale : float
        Compensation factor (total / num_sampled) intended to be
        multiplied onto the event weight branch so that the SUM of
        weights in the subsample remains an unbiased estimator of the
        full-sample weight sum.
    """
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
        log.info(
            f"  requested {num} events but only {total} available "
            f"-- using all events"
        )

    return arrays, weight_scale


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

@tensorclass
class WWbbData(UnfoldingData):

    @classmethod
    def read(
        cls,
        path_sim: Union[str, List[str]],
        path_data: Union[str, List[str]],
        max_jets: int = 6,
        device: Optional[torch.device] = None,
        num_sim: Optional[int] = None,
        num_data: Optional[int] = None,
        num: Optional[int] = None,
        normalize: bool = False,
        seed: int = 42,
        include_tagging: bool = True,
    ):
        """
        path_data may point at either Herwig pseudodata (MC, has truth +
        weight_total_NOSYS) or real collision data (has neither).
        Availability is detected automatically per population by
        inspecting the parquet schema.
        """
        if num is not None:
            if num_sim is None:
                num_sim = num
            if num_data is None:
                num_data = num

        batch_size = 0
        tensor_kwargs = defaultdict(list)
        n_tokens = 2 + max_jets
        raw_weights = []

        for i, (path, num_i) in enumerate(
            ((path_sim, num_sim), (path_data, num_data))
        ):
            paths = _resolve_paths(path)
            label = "sim" if i == 0 else "data"
            log.info(f"[{label}] reading {len(paths)} parquet file(s) from {path}")

            available = _available_columns(paths)
            has_weight = RECO_BRANCHES["weight"] in available
            has_truth = all(b in available for b in PARTICLE_BRANCHES.values())
            log.info(f"[{label}] has_weight={has_weight}, has_truth={has_truth}")

            columns = _needed_columns(RECO_BRANCHES, include_weight=has_weight)
            if has_truth:
                columns += _needed_columns(PARTICLE_BRANCHES)
            columns = list(dict.fromkeys(columns))

            arrays, weight_scale = _load_arrays(paths, columns, num=num_i, seed=seed)
            n_events = len(arrays)
            log.info(f"[{label}] loaded {n_events} events")

            # ---------------- RECO LEVEL ----------------
            features_x, mask_x, _ = _build_tokens(
                arrays, RECO_BRANCHES, max_jets, n_events,
                include_tagging=include_tagging, drop_invalid=False,
            )

            # ---------------- PARTICLE LEVEL ----------------
            if has_truth:
                features_z, mask_z, valid_z = _build_tokens(
                    arrays, PARTICLE_BRANCHES, max_jets, n_events,
                    include_tagging=False, drop_invalid=True,
                )
                keep_mask = valid_z
            else:
                features_z = torch.zeros((n_events, n_tokens, 4 + 3))  # 4-vec + is_mu/is_met/is_jet
                mask_z = torch.zeros((n_events, n_tokens), dtype=torch.bool)
                keep_mask = torch.ones(n_events, dtype=torch.bool)

            n_kept = keep_mask.sum().item()
            if n_kept < n_events:
                log.info(
                    f"[{label}] dropping {n_events - n_kept} events "
                    f"with invalid truth muon"
                )

            features_x = features_x[keep_mask]
            mask_x = mask_x[keep_mask]
            features_z = torch.nan_to_num(features_z[keep_mask], nan=0.0)
            mask_z = mask_z[keep_mask]

            tensor_kwargs["x"].append(features_x)
            tensor_kwargs["mask_x"].append(mask_x)
            tensor_kwargs["z"].append(features_z)
            tensor_kwargs["mask_z"].append(mask_z)

            if has_weight:
                w = ak.to_numpy(arrays[RECO_BRANCHES["weight"]]).astype(np.float32)
            else:
                w = np.ones(n_events, dtype=np.float32)
            w = w[keep_mask.numpy()]
            w = w * weight_scale
            raw_weights.append(w)

            n_events = n_kept
            tensor_kwargs["labels"].append(
                torch.full((n_events,), i, dtype=torch.float32)
            )

            batch_size += n_events

        if normalize:
            w_sim_raw, w_data_raw = raw_weights[0], raw_weights[1]
            sum_sim  = np.abs(w_sim_raw).sum()
            sum_data = np.abs(w_data_raw).sum()
            target   = batch_size / 2.0
            scale_sim  = target / sum_sim
            scale_data = target / sum_data

            log.info(
                f"Normalizing weights: "
                f"sim scale={scale_sim:.4f} "
                f"(sum: {sum_sim:.1f} -> {sum_sim * scale_sim:.1f}), "
                f"data scale={scale_data:.4f} "
                f"(sum: {sum_data:.1f} -> {sum_data * scale_data:.1f})"
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


@tensorclass
class WWbbTransform:
    """
    Preprocessing for the token-based WWbb features with a minimal
    scalar block: [E, px, py, pz, is_mu, is_met, is_jet].

    Applies a GLOBAL EQUIVARIANT RESCALE of the four-momenta
    (E, px, py, pz) by a single scalar `scale_p`. Dividing all four
    components by the SAME constant preserves Lorentz-covariance
    exactly (a Lorentz boost/rotation commutes with scalar
    multiplication: Lambda(x / s) = (Lambda x) / s), while bringing raw
    GeV-scale momenta down to an O(1) numerical range -- matching the
    O(1) scale of the fixed beam spurions injected by LorentzTransform.

    The one-hot type flags (is_mu, is_met, is_jet) are already {0, 1}
    and need no further standardization.

    NOTE: scale_p below is a PLACEHOLDER. Run compute_shift_scale() once
    on your sim data and paste the printed value into WWbbProcess before
    training.
    """

    scale_p: float = 50.0

    def forward(self, batch):

        # global equivariant rescale of four-momenta
        batch.x[..., :4] = batch.x[..., :4] / self.scale_p
        batch.z[..., :4] = batch.z[..., :4] / self.scale_p

        batch.x, batch.mask_x = LorentzTransform.forward(batch.x, mask=batch.mask_x)
        batch.z, batch.mask_z = LorentzTransform.forward(batch.z, mask=batch.mask_z)

        return batch

    def reverse(self, batch):
        raise NotImplementedError

# ----------------------------------------------------------------------------
# Utility: compute scale_p / shift / scale from your sim data
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Utility: compute scale_p from your sim data
# ----------------------------------------------------------------------------

def compute_shift_scale(
    path_sim: str,
    max_jets: int = 3,
    num: Optional[int] = None,
    seed: int = 42,
):
    """
    Compute, from the sim parquet files, the RMS four-momentum-component
    magnitude over all valid (non-padded) tokens -- used as the global
    equivariant rescale `scale_p` of (E, px, py, pz).

    Usage:
        python -c "
        from src.datasets.wwbb import compute_shift_scale
        compute_shift_scale('/scratch/mjosef/Unfolding/aussie/data/sim', num=2_000_000)
        "
    """
    paths = _resolve_paths(path_sim)
    available = _available_columns(paths)
    has_truth = all(b in available for b in PARTICLE_BRANCHES.values())
    assert has_truth, "compute_shift_scale requires a sim sample with particle-level truth"

    columns = _needed_columns(RECO_BRANCHES, include_weight=False)
    columns += _needed_columns(PARTICLE_BRANCHES)
    columns = list(dict.fromkeys(columns))

    arrays, _ = _load_arrays(paths, columns, num=num, seed=seed)
    n_events = len(arrays)

    features_x, mask_x, _ = _build_tokens(
        arrays, RECO_BRANCHES, max_jets, n_events, drop_invalid=False,
    )
    features_z, mask_z, valid_z = _build_tokens(
        arrays, PARTICLE_BRANCHES, max_jets, n_events, drop_invalid=True,
    )

    n_dropped = (~valid_z).sum().item()
    if n_dropped > 0:
        print(f"Dropping {n_dropped} / {n_events} events with invalid truth muon "
              f"before computing statistics")

    features_x = features_x[valid_z]
    mask_x = mask_x[valid_z]
    features_z = torch.nan_to_num(features_z[valid_z], nan=0.0)
    mask_z = mask_z[valid_z]

    vec_x = features_x[..., :4][mask_x]
    vec_z = features_z[..., :4][mask_z]
    scale_p = torch.sqrt(torch.cat([vec_x, vec_z]).pow(2).mean()).item()

    print("Paste into WWbbProcess:\n")
    print(f"  scale_p={scale_p:.4f},")

    return scale_p

@dataclass
class WWbbProcess:
    num_features: int = 12   # reco: 4 (4-vec) + 8 (scalars); NOTE: not used
                              # for functional network sizing here -- L-GATr
                              # in_v_channels/in_s_channels are set explicitly
                              # in the model config, unlike WWbbMultiProcess
                              # where this field drives the MLP's dim_in via
                              # interpolation
    transforms: Tuple[Callable] = (
        WWbbTransform(
            scale_p=226.4290,
        ),
    )

    observables_x: Tuple[Observable] = (
        # ================================================================
        # BENCHMARK CROSS-CHECK OBSERVABLES -- identical name/definition/
        # binning to WWbbMultiProcess's observables_x, so observables.pdf/
        # metrics.pdf/iteration chi2-summary can be compared page-for-page
        # between the L-GATr (wwbb) and MLP+Kernel (wwbb_multi) pipelines.
        # ================================================================
        Observable(
            name="mu_pt",
            compute=lambda x: torch.sqrt(x[..., 0, 1] ** 2 + x[..., 0, 2] ** 2),
            label=r"$p_{T,\mu}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), logy=True, log_bins=True,
        ),
        Observable(
            name="mu_e",
            compute=lambda x: x[..., 0, 0],
            label=r"$E_{\mu}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), logy=True, log_bins=True,
        ),
        Observable(
            name="j1_pt",
            compute=lambda x: torch.sqrt(x[..., 2, 1] ** 2 + x[..., 2, 2] ** 2),
            label=r"Leading jet $p_T$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), logy=True, log_bins=True,
        ),
        Observable(
            name="j1_e",
            compute=lambda x: x[..., 2, 0],
            label=r"Leading jet $E$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), logy=True, log_bins=True,
        ),
        Observable(
            name="j2_pt",
            compute=lambda x: torch.sqrt(x[..., 3, 1] ** 2 + x[..., 3, 2] ** 2),
            label=r"Subleading jet $p_T$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), logy=True, log_bins=True,
        ),
        Observable(
            name="j2_e",
            compute=lambda x: x[..., 3, 0],
            label=r"Subleading jet $E$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), logy=True, log_bins=True,
        ),
        Observable(
            name="j3_pt",
            compute=lambda x: torch.sqrt(x[..., 4, 1] ** 2 + x[..., 4, 2] ** 2),
            label=r"3rd jet $p_T$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), logy=True, log_bins=True,
        ),
        Observable(
            name="j3_e",
            compute=lambda x: x[..., 4, 0],
            label=r"3rd jet $E$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), logy=True, log_bins=True,
        ),

        # ================================================================
        # EXISTING DIAGNOSTIC OBSERVABLES
        # ================================================================
        Observable(
            name="mu_eta",
            compute=lambda x: torch.arctanh(
                x[..., 0, 3] / torch.sqrt(x[..., 0, 1] ** 2 + x[..., 0, 2] ** 2 + x[..., 0, 3] ** 2).clamp(min=1e-8)
            ),
            label=r"$\eta_{\mu}$",
            xlims=(-5, 5), log_bins=False,
        ),
        Observable(
            name="j1_eta",
            compute=lambda x: torch.arctanh(
                x[..., 2, 3] / torch.sqrt(x[..., 2, 1] ** 2 + x[..., 2, 2] ** 2 + x[..., 2, 3] ** 2).clamp(min=1e-8)
            ),
            label=r"Leading jet $\eta$",
            xlims=(-5, 5), log_bins=False,
        ),
        Observable(
            name="j2_eta",
            compute=lambda x: torch.arctanh(
                x[..., 3, 3] / torch.sqrt(x[..., 3, 1] ** 2 + x[..., 3, 2] ** 2 + x[..., 3, 3] ** 2).clamp(min=1e-8)
            ),
            label=r"Subleading jet $\eta$",
            xlims=(-5, 5), log_bins=False,
        ),
        Observable(
            name="j3_eta",
            compute=lambda x: torch.arctanh(
                x[..., 4, 3] / torch.sqrt(x[..., 4, 1] ** 2 + x[..., 4, 2] ** 2 + x[..., 4, 3] ** 2).clamp(min=1e-8)
            ),
            label=r"3rd jet $\eta$",
            xlims=(-5, 5), log_bins=False,
        ),
        Observable(
            name="met",
            compute=lambda x: x[..., 1, 0],
            label=r"$E_T^{\text{miss}}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), log_bins=True,
        ),
    )

    observables_z: Tuple[Observable] = (
        # ================================================================
        # BENCHMARK CROSS-CHECK OBSERVABLES (truth level)
        # ================================================================
        Observable(
            name="mu_pt_truth",
            compute=lambda z: torch.sqrt(z[..., 0, 1] ** 2 + z[..., 0, 2] ** 2),
            label=r"$p_{T,\mu}^{\rm truth}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), logy=True, log_bins=True,
        ),
        Observable(
            name="mu_e_truth",
            compute=lambda z: z[..., 0, 0],
            label=r"$E_{\mu}^{\rm truth}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), logy=True, log_bins=True,
        ),
        Observable(
            name="j1_pt_truth",
            compute=lambda z: torch.sqrt(z[..., 2, 1] ** 2 + z[..., 2, 2] ** 2),
            label=r"Leading jet $p_T^{\rm truth}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), logy=True, log_bins=True,
        ),
        Observable(
            name="j1_e_truth",
            compute=lambda z: z[..., 2, 0],
            label=r"Leading jet $E^{\rm truth}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), logy=True, log_bins=True,
        ),
        Observable(
            name="j2_pt_truth",
            compute=lambda z: torch.sqrt(z[..., 3, 1] ** 2 + z[..., 3, 2] ** 2),
            label=r"Subleading jet $p_T^{\rm truth}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), logy=True, log_bins=True,
        ),
        Observable(
            name="j2_e_truth",
            compute=lambda z: z[..., 3, 0],
            label=r"Subleading jet $E^{\rm truth}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), logy=True, log_bins=True,
        ),
        Observable(
            name="j3_pt_truth",
            compute=lambda z: torch.sqrt(z[..., 4, 1] ** 2 + z[..., 4, 2] ** 2),
            label=r"3rd jet $p_T^{\rm truth}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), logy=True, log_bins=True,
        ),
        Observable(
            name="j3_e_truth",
            compute=lambda z: z[..., 4, 0],
            label=r"3rd jet $E^{\rm truth}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), logy=True, log_bins=True,
        ),

        # ================================================================
        # EXISTING DIAGNOSTIC OBSERVABLES (truth level)
        # ================================================================
        Observable(
            name="mu_eta_truth",
            compute=lambda z: torch.arctanh(
                z[..., 0, 3] / torch.sqrt(z[..., 0, 1] ** 2 + z[..., 0, 2] ** 2 + z[..., 0, 3] ** 2).clamp(min=1e-8)
            ),
            label=r"$\eta_{\mu}^{\text{truth}}$",
            xlims=(-5, 5), log_bins=False,
        ),
        Observable(
            name="j1_eta_truth",
            compute=lambda z: torch.arctanh(
                z[..., 2, 3] / torch.sqrt(z[..., 2, 1] ** 2 + z[..., 2, 2] ** 2 + z[..., 2, 3] ** 2).clamp(min=1e-8)
            ),
            label=r"Leading jet $\eta^{\text{truth}}$",
            xlims=(-5, 5), log_bins=False,
        ),
        Observable(
            name="j2_eta_truth",
            compute=lambda z: torch.arctanh(
                z[..., 3, 3] / torch.sqrt(z[..., 3, 1] ** 2 + z[..., 3, 2] ** 2 + z[..., 3, 3] ** 2).clamp(min=1e-8)
            ),
            label=r"Subleading jet $\eta^{\text{truth}}$",
            xlims=(-5, 5), log_bins=False,
        ),
        Observable(
            name="j3_eta_truth",
            compute=lambda z: torch.arctanh(
                z[..., 4, 3] / torch.sqrt(z[..., 4, 1] ** 2 + z[..., 4, 2] ** 2 + z[..., 4, 3] ** 2).clamp(min=1e-8)
            ),
            label=r"3rd jet $\eta^{\text{truth}}$",
            xlims=(-5, 5), log_bins=False,
        ),
        Observable(
            name="met_truth",
            compute=lambda z: z[..., 1, 0],
            label=r"$E_T^{\text{miss,truth}}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3), log_bins=True,
        ),
    )