import glob
import logging
import os

import awkward as ak
import numpy as np
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

    This is what lets WWbbData.read() be pointed directly at e.g.
    '.../Run3/sim' or '.../Run3/pseudodata' without listing files by hand.
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


def _build_tokens(arrays, branch_map, max_jets, n_events, include_tagging):
    """Build (features, mask) for one event collection at one level (reco/truth).

    Token layout: [muon, met, jet_0, ..., jet_{max_jets-1}]
    Feature layout: [E, px, py, pz, is_mu, is_met, is_jet, (tagging scores...)]

    All pt/energy quantities are converted from MeV (raw ntuple units) to GeV.
    """
    n_tokens = 2 + max_jets

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

    if include_tagging:
        pb   = _pad(arrays[branch_map["jet_gn2_pb"]], max_jets)
        pc   = _pad(arrays[branch_map["jet_gn2_pc"]], max_jets)
        pu   = _pad(arrays[branch_map["jet_gn2_pu"]], max_jets)
        ptau = _pad(arrays[branch_map["jet_gn2_ptau"]], max_jets)

        n_scalars = 3 + N_JET_TAGGING_SCORES
        scalars = np.zeros((n_events, n_tokens, n_scalars), dtype=np.float32)
        scalars[:, 0, 0] = 1.0
        scalars[:, 1, 1] = 1.0
        scalars[:, 2:, 2] = 1.0
        scalars[:, 2:, 3] = pb
        scalars[:, 2:, 4] = pc
        scalars[:, 2:, 5] = pu
        scalars[:, 2:, 6] = ptau
    else:
        n_scalars = 3
        scalars = np.zeros((n_events, n_tokens, n_scalars), dtype=np.float32)
        scalars[:, 0, 0] = 1.0
        scalars[:, 1, 1] = 1.0
        scalars[:, 2:, 2] = 1.0

    features = np.concatenate([momenta, scalars], axis=-1)

    mask = np.zeros((n_events, n_tokens), dtype=bool)
    mask[:, :2] = True
    mask[:, 2:] = np.arange(max_jets)[None, :] < n_jets[:, None]

    return torch.from_numpy(features).float(), torch.from_numpy(mask)


def _needed_columns(branch_map, include_weight=False):
    cols = [v for k, v in branch_map.items() if k != "weight"]
    if include_weight and "weight" in branch_map:
        cols.append(branch_map["weight"])
    return cols


def _load_arrays(paths, columns, num=None):
    """Load one or more parquet files as a single Awkward Array,
    optionally truncating to the first `num` events overall."""
    if len(paths) == 1:
        arrays = ak.from_parquet(paths[0], columns=columns)
    else:
        arrays = ak.concatenate(
            [ak.from_parquet(p, columns=columns) for p in paths]
        )
    if num is not None:
        arrays = arrays[:num]
    return arrays


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
        max_jets: int = 10,
        device: Optional[torch.device] = None,
        num_sim: Optional[int] = None,
        num_data: Optional[int] = None,
        num: Optional[int] = None,  
    ):
        if num is not None:
            if num_sim is None:
                num_sim = num
            if num_data is None:
                num_data = num

        batch_size = 0
        tensor_kwargs = defaultdict(list)
        n_tokens = 2 + max_jets

        for i, (path, num) in enumerate(
            ((path_sim, num_sim), (path_data, num_data))
        ):
            paths = _resolve_paths(path)
            label = "sim" if i == 0 else "pseudodata"
            log.info(f"[{label}] reading {len(paths)} parquet file(s) from {path}")

            # both sim AND pseudodata have particle-level branches
            # (Herwig is MC, so it has full truth information just like Pythia)
            columns = _needed_columns(RECO_BRANCHES, include_weight=True)
            columns += _needed_columns(PARTICLE_BRANCHES)
            columns = list(dict.fromkeys(columns))

            arrays = _load_arrays(paths, columns, num=num)
            n_events = len(arrays)
            log.info(f"[{label}] loaded {n_events} events")

            # ---------------- RECO LEVEL ----------------
            features_x, mask_x = _build_tokens(
                arrays, RECO_BRANCHES, max_jets, n_events, include_tagging=True,
            )
            tensor_kwargs["x"].append(features_x)
            tensor_kwargs["mask_x"].append(mask_x)

            # event weights
            weight_branch = RECO_BRANCHES["weight"]
            if weight_branch in arrays.fields:
                w = ak.to_numpy(arrays[weight_branch]).astype(np.float32)
            else:
                w = np.ones(n_events, dtype=np.float32)
            logw = np.log(np.abs(w) + 1e-12)
            tensor_kwargs["sample_logweights"].append(torch.from_numpy(logw).float())

            # ---------------- PARTICLE LEVEL (both sim and pseudodata) ----------------
            # Herwig pseudodata is MC -- it has full particle-level truth information
            # just like Pythia sim. We read it for both populations so that
            # classification.py's plot() can compare z_sim vs z_dat correctly,
            # exactly mirroring the wwbb_multi.py approach.
            features_z, mask_z = _build_tokens(
                arrays, PARTICLE_BRANCHES, max_jets, n_events, include_tagging=False,
            )
            tensor_kwargs["z"].append(features_z)
            tensor_kwargs["mask_z"].append(mask_z)

            tensor_kwargs["labels"].append(
                torch.full((n_events,), i, dtype=torch.float32)
            )

            batch_size += n_events

        for k in tensor_kwargs:
            tensor_kwargs[k] = torch.cat(tensor_kwargs[k], dim=0)

        return cls(batch_size=[batch_size], device=device, **tensor_kwargs)


@tensorclass
class WWbbTransform:
    """Logit-transform GN2 tagging scores, then prepend Lorentz spurions."""

    eps: float = 1e-6

    def forward(self, batch):
        is_jet = batch.x[..., 6:7]
        scores = batch.x[..., 7:11].clamp(self.eps, 1 - self.eps)
        logit_scores = torch.log(scores / (1 - scores))
        batch.x[..., 7:11] = logit_scores * is_jet

        batch.x, batch.mask_x = LorentzTransform.forward(batch.x, mask=batch.mask_x)
        batch.z, batch.mask_z = LorentzTransform.forward(batch.z, mask=batch.mask_z)

        return batch

    def reverse(self, batch):
        raise NotImplementedError


@dataclass
class WWbbProcess:
    num_features: int = 11
    transforms: Tuple[Callable] = (WWbbTransform(),)

    observables_x: Tuple[Observable] = (
        # ---- muon ----
        Observable(
            name="mu_pt",
            compute=lambda x: torch.sqrt(x[..., 0, 1] ** 2 + x[..., 0, 2] ** 2),
            label=r"$p_{T,\mu}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="mu_eta",
            compute=lambda x: torch.arctanh(
                x[..., 0, 3] / torch.sqrt(x[..., 0, 1] ** 2 + x[..., 0, 2] ** 2 + x[..., 0, 3] ** 2).clamp(min=1e-8)
            ),
            label=r"$\eta_{\mu}$",
            xlims=(-5, 5),
        ),
        Observable(
            name="mu_e",
            compute=lambda x: x[..., 0, 0],
            label=r"$E_{\mu}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
        # ---- leading jet ----
        Observable(
            name="j1_pt",
            compute=lambda x: torch.sqrt(x[..., 2, 1] ** 2 + x[..., 2, 2] ** 2),
            label=r"Leading jet $p_T$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="j1_eta",
            compute=lambda x: torch.arctanh(
                x[..., 2, 3] / torch.sqrt(x[..., 2, 1] ** 2 + x[..., 2, 2] ** 2 + x[..., 2, 3] ** 2).clamp(min=1e-8)
            ),
            label=r"Leading jet $\eta$",
            xlims=(-5, 5),
        ),
        Observable(
            name="j1_e",
            compute=lambda x: x[..., 2, 0],
            label=r"Leading jet $E$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
        # ---- subleading jet ----
        Observable(
            name="j2_pt",
            compute=lambda x: torch.sqrt(x[..., 3, 1] ** 2 + x[..., 3, 2] ** 2),
            label=r"Subleading jet $p_T$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="j2_eta",
            compute=lambda x: torch.arctanh(
                x[..., 3, 3] / torch.sqrt(x[..., 3, 1] ** 2 + x[..., 3, 2] ** 2 + x[..., 3, 3] ** 2).clamp(min=1e-8)
            ),
            label=r"Subleading jet $\eta$",
            xlims=(-5, 5),
        ),
        Observable(
            name="j2_e",
            compute=lambda x: x[..., 3, 0],
            label=r"Subleading jet $E$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
        # ---- third leading jet ----
        Observable(
            name="j3_pt",
            compute=lambda x: torch.sqrt(x[..., 4, 1] ** 2 + x[..., 4, 2] ** 2),
            label=r"3rd jet $p_T$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="j3_eta",
            compute=lambda x: torch.arctanh(
                x[..., 4, 3] / torch.sqrt(x[..., 4, 1] ** 2 + x[..., 4, 2] ** 2 + x[..., 4, 3] ** 2).clamp(min=1e-8)
            ),
            label=r"3rd jet $\eta$",
            xlims=(-5, 5),
        ),
        Observable(
            name="j3_e",
            compute=lambda x: x[..., 4, 0],
            label=r"3rd jet $E$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
        # ---- MET (kept as a useful cross-check) ----
        Observable(
            name="met",
            compute=lambda x: x[..., 1, 0],
            label=r"$E_T^{\text{miss}}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
    )

    observables_z: Tuple[Observable] = (
        # ---- muon (truth) ----
        Observable(
            name="mu_pt_truth",
            compute=lambda z: torch.sqrt(z[..., 0, 1] ** 2 + z[..., 0, 2] ** 2),
            label=r"$p_{T,\mu}^{\text{truth}}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="mu_eta_truth",
            compute=lambda z: torch.arctanh(
                z[..., 0, 3] / torch.sqrt(z[..., 0, 1] ** 2 + z[..., 0, 2] ** 2 + z[..., 0, 3] ** 2).clamp(min=1e-8)
            ),
            label=r"$\eta_{\mu}^{\text{truth}}$",
            xlims=(-5, 5),
        ),
        Observable(
            name="mu_e_truth",
            compute=lambda z: z[..., 0, 0],
            label=r"$E_{\mu}^{\text{truth}}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
        # ---- leading jet (truth) ----
        Observable(
            name="j1_pt_truth",
            compute=lambda z: torch.sqrt(z[..., 2, 1] ** 2 + z[..., 2, 2] ** 2),
            label=r"Leading jet $p_T^{\text{truth}}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="j1_eta_truth",
            compute=lambda z: torch.arctanh(
                z[..., 2, 3] / torch.sqrt(z[..., 2, 1] ** 2 + z[..., 2, 2] ** 2 + z[..., 2, 3] ** 2).clamp(min=1e-8)
            ),
            label=r"Leading jet $\eta^{\text{truth}}$",
            xlims=(-5, 5),
        ),
        Observable(
            name="j1_e_truth",
            compute=lambda z: z[..., 2, 0],
            label=r"Leading jet $E^{\text{truth}}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
        # ---- subleading jet (truth) ----
        Observable(
            name="j2_pt_truth",
            compute=lambda z: torch.sqrt(z[..., 3, 1] ** 2 + z[..., 3, 2] ** 2),
            label=r"Subleading jet $p_T^{\text{truth}}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="j2_eta_truth",
            compute=lambda z: torch.arctanh(
                z[..., 3, 3] / torch.sqrt(z[..., 3, 1] ** 2 + z[..., 3, 2] ** 2 + z[..., 3, 3] ** 2).clamp(min=1e-8)
            ),
            label=r"Subleading jet $\eta^{\text{truth}}$",
            xlims=(-5, 5),
        ),
        Observable(
            name="j2_e_truth",
            compute=lambda z: z[..., 3, 0],
            label=r"Subleading jet $E^{\text{truth}}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
        # ---- third leading jet (truth) ----
        Observable(
            name="j3_pt_truth",
            compute=lambda z: torch.sqrt(z[..., 4, 1] ** 2 + z[..., 4, 2] ** 2),
            label=r"3rd jet $p_T^{\text{truth}}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="j3_eta_truth",
            compute=lambda z: torch.arctanh(
                z[..., 4, 3] / torch.sqrt(z[..., 4, 1] ** 2 + z[..., 4, 2] ** 2 + z[..., 4, 3] ** 2).clamp(min=1e-8)
            ),
            label=r"3rd jet $\eta^{\text{truth}}$",
            xlims=(-5, 5),
        ),
        Observable(
            name="j3_e_truth",
            compute=lambda z: z[..., 4, 0],
            label=r"3rd jet $E^{\text{truth}}$ [GeV]",
            xlims=(1e-3, 1 - 1e-3),
        ),
        # ---- MET (truth) ----
        Observable(
            name="met_truth",
            compute=lambda z: z[..., 1, 0],
            label=r"$E_T^{\text{miss,truth}}$ [GeV]",
            qlims=(1e-3, 1 - 1e-3),
        ),
    )