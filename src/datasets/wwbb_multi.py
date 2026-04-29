import numpy as np
import torch
import h5py
import os

from collections import defaultdict
from dataclasses import dataclass
from tensordict import tensorclass
from typing import Callable, Tuple, Optional

from src.datasets.base_dataset import UnfoldingData
from src.utils.observable import Observable
from src.utils.transforms import ShiftAndScale, LogScale


@tensorclass
class WWbbMultiData(UnfoldingData):
    """
    Scalar-feature dataset for WWbb -> lepton + jets, MultiFold style.

    6 scalar pT features extracted from the full [N, 12, F] particle array:

        Index 0 : pT lepton
        Index 1 : pT b-jet 1      (leading)
        Index 2 : pT b-jet 2      (sub-leading)
        Index 3 : pT light jet 1  (leading)
        Index 4 : pT light jet 2  (sub-leading)
        Index 5 : pT MET

    Particle ordering in the h5 file (fixed 12 slots):
        0  lepton
        1  b-jet 1
        2  b-jet 2
        3  b-jet 3
        4  b-jet 4
        5  light jet 1
        6  light jet 2
        7  light jet 3
        8  light jet 4
        9  light jet 5
        10 light jet 6
        11 MET

    H5 file structure:
        file['reco']:    [#events, 12, #features]           always present
        file['gen']:     [#events, #particles, #features]   OPTIONAL
        file['weights']: [#events]                          always present

    NOTE: Real (observed) data never has a gen level. When absent,
          'z' is zero-padded so shapes stay consistent.
    """

    @classmethod
    def read(
        cls,
        path: str,
        num: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Read WWbb data from two h5 files found under `path`.

        Expected filenames:
            mc   : df_pythia_ttbar_train.h5
            data : df_herwig_ttbar_singletop_DR_train.h5

        Args:
            path   : Directory containing both h5 files.
            num    : Optional max number of events to load per file.
            device : Torch device to place tensors on.
        """

        # label=0 -> mc (simulated), label=1 -> data (observed)
        # mc first so gen_shape is known before data (which may lack gen)
        file_cfg = [
            (
                "mc",
                os.path.join(path, "df_pythia_ttbar_train.h5"),
                0,
            ),
            (
                "data",
                os.path.join(path, "df_herwig_ttbar_singletop_DR_train.h5"),
                #os.path.join(path, "data.h5"),
                1,
            ),
        ]

        tensor_kwargs = defaultdict(list)
        batch_size    = 0

        for split_name, fpath, label_val in file_cfg:
            with h5py.File(fpath, "r") as f:

                has_gen = "gen" in f

                if split_name == "data" and not has_gen:
                    print(
                        f"[{split_name}] No 'gen' dataset found in {fpath}.\n"
                        "  -> Expected for real (observed) data. "
                        "z will be zero-padded."
                    )
                elif split_name == "mc" and not has_gen:
                    print(
                        f"[{split_name}] WARNING: No 'gen' dataset found in "
                        f"MC file {fpath}. Cross-checks will not be possible."
                    )

                reco_raw    = f["reco"]   [:num].astype(np.float32)  # [N, 12, F]
                weights_raw = f["weights"][:num].astype(np.float32)  # [N]
                gen_raw     = (
                    f["gen"][:num].astype(np.float32) if has_gen else None
                )

                print(
                    f"[{split_name}] "
                    f"reco: {reco_raw.shape}  |  "
                    f"gen: {gen_raw.shape if gen_raw is not None else 'N/A'}  |  "
                    f"weights: {weights_raw.shape}"
                )

            # --- extract 6 scalar pT features ----------------------------
            x = cls._extract_pt_features(reco_raw)   # [N, 6]
            z = (
                cls._extract_pt_features(gen_raw)
                if gen_raw is not None
                else torch.zeros(reco_raw.shape[0], 6, dtype=torch.float32)
            )

            n_events = x.shape[0]
            labels   = torch.full(
                [n_events], float(label_val), dtype=torch.float32
            )

            tensor_kwargs["x"].append(x)
            tensor_kwargs["z"].append(z)
            tensor_kwargs["labels"].append(labels)

            # sample_logweights: zero = uniform (matches ZJetData convention)
            w = torch.from_numpy(weights_raw).float().clamp(min=1e-10)
            tensor_kwargs["sample_logweights"].append(w.log())

            batch_size += n_events

        for k in tensor_kwargs:
            tensor_kwargs[k] = torch.cat(tensor_kwargs[k], dim=0)

        return cls(
            batch_size=[batch_size],
            device=device,
            **tensor_kwargs,
        )

    @staticmethod
    def _extract_pt_features(raw: np.ndarray) -> torch.Tensor:
        """
        Extract 6 scalar pT features from the full particle array.

        Parameters
        ----------
        raw : np.ndarray  [N, n_particles, n_features]
            Feature index 0 = pt for every particle slot.

        Returns
        -------
        torch.Tensor  [N, 6]
            [pT_lepton, pT_b1, pT_b2, pT_j1, pT_j2, pT_MET]
        """
        features = np.stack([
            raw[:, 0,  0],   # lepton       slot 0,  feature 0
            raw[:, 1,  0],   # b-jet 1      slot 1,  feature 0
            raw[:, 2,  0],   # b-jet 2      slot 2,  feature 0
            raw[:, 5,  0],   # light jet 1  slot 5,  feature 0
            raw[:, 6,  0],   # light jet 2  slot 6,  feature 0
            raw[:, 11, 0],   # MET          slot 11, feature 0
        ], axis=1)           # [N, 6]

        return torch.from_numpy(features).float()


# ---------------------------------------------------------------------------
# Transform  — uses ShiftAndScale and LogScale from transforms.py directly
# ---------------------------------------------------------------------------

@tensorclass
class WWbbMultiTransform:
    """
    Preprocessing for the 6 scalar pT features.

    Pipeline (mirrors ZJetTransform style):

        1. LogScale.forward  : x -> log(x + eps)
                               Applied to all 6 features (all are pT values
                               so all benefit from log compression).

        2. ShiftAndScale.forward : x -> (x - shift) / scale
                               Standardises each feature to ~ N(0, 1).

    Reverse pipeline:
        1. ShiftAndScale.reverse : x -> x * scale + shift
        2. LogScale.reverse      : x -> exp(x)   (eps already baked in)

    Parameters
    ----------
    shift_x, shift_z : torch.Tensor [6]
        Per-feature mean after log transform (reco / gen level).
    scale_x, scale_z : torch.Tensor [6]
        Per-feature std  after log transform (reco / gen level).
    eps : float
        Small offset added before log to avoid log(0).

    NOTE: The default shift/scale values are PLACEHOLDERS (mean=0, std=1).
          Run compute_shift_scale() once on your MC file and paste the
          printed values here before training.
    """

    shift_x: torch.Tensor   # [6]
    shift_z: torch.Tensor   # [6]
    scale_x: torch.Tensor   # [6]
    scale_z: torch.Tensor   # [6]
    eps: float = 1e-3

    # all 6 features are pT values -> apply log to every index
    _log_indices: tuple = (0, 1, 2, 3, 4, 5)

    def forward(self, batch):

        # 1. log-compress all pT features using LogScale from transforms.py
        batch.x = LogScale.forward(batch.x, indices=list(self._log_indices), eps=self.eps)
        batch.z = LogScale.forward(batch.z, indices=list(self._log_indices), eps=self.eps)

        # 2. standardise using ShiftAndScale from transforms.py
        batch.x = ShiftAndScale.forward(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.z = ShiftAndScale.forward(batch.z, shift=self.shift_z, scale=self.scale_z)

        return batch

    def reverse(self, batch):

        # 1. undo standardisation
        batch.x = ShiftAndScale.reverse(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.z = ShiftAndScale.reverse(batch.z, shift=self.shift_z, scale=self.scale_z)

        # 2. undo log
        batch.x = LogScale.reverse(batch.x, indices=list(self._log_indices))
        batch.z = LogScale.reverse(batch.z, indices=list(self._log_indices))

        return batch


# ---------------------------------------------------------------------------
# Utility: compute shift / scale from your MC file
# ---------------------------------------------------------------------------

def compute_shift_scale(path_mc: str, num: Optional[int] = None):
    """
    Compute per-feature mean and std after log transform from the MC file.

    Run once, then paste the printed tensors into WWbbMultiProcess.

    Usage
    -----
    From the command line:

        python -c "
        from src.datasets.wwbb_multi import compute_shift_scale
        compute_shift_scale(
            '/ptmp/mpp/mjosef/data_files/WbWb_files/'
            'bulk_region/h5_files/df_pythia_ttbar_train.h5'
        )
        "

    Or from a notebook / script:

        from src.datasets.wwbb_multi import compute_shift_scale
        shift_x, scale_x, shift_z, scale_z = compute_shift_scale(path_mc)
    """
    eps = 1e-3
    log_idx = [0, 1, 2, 3, 4, 5]

    with h5py.File(path_mc, "r") as f:
        reco_raw = f["reco"][:num].astype(np.float32)
        gen_raw  = f["gen"] [:num].astype(np.float32)

    x = WWbbMultiData._extract_pt_features(reco_raw)
    z = WWbbMultiData._extract_pt_features(gen_raw)

    # apply log exactly as the transform does
    x = LogScale.forward(x, indices=log_idx, eps=eps)
    z = LogScale.forward(z, indices=log_idx, eps=eps)

    shift_x, scale_x = x.mean(0), x.std(0)
    shift_z, scale_z = z.mean(0), z.std(0)

    print("Paste into WWbbMultiProcess:\n")
    print(f"  shift_x=torch.tensor({[round(v, 4) for v in shift_x.tolist()]}),")
    print(f"  scale_x=torch.tensor({[round(v, 4) for v in scale_x.tolist()]}),")
    print(f"  shift_z=torch.tensor({[round(v, 4) for v in shift_z.tolist()]}),")
    print(f"  scale_z=torch.tensor({[round(v, 4) for v in scale_z.tolist()]}),")

    return shift_x, scale_x, shift_z, scale_z


# ---------------------------------------------------------------------------
# Process descriptor
# ---------------------------------------------------------------------------

@dataclass
class WWbbMultiProcess:
    """
    Process descriptor for WWbb MultiFold with 6 scalar pT features.

    Feature ordering (reco and gen identical):
        0 : pT lepton
        1 : pT b-jet 1         (leading)
        2 : pT b-jet 2         (sub-leading)
        3 : pT light jet 1     (leading)
        4 : pT light jet 2     (sub-leading)
        5 : pT MET
    """

    num_features: int = 6

    transforms: Tuple[Callable] = (
        WWbbMultiTransform(
            # ----------------------------------------------------------------
            # PLACEHOLDERS — replace with output of compute_shift_scale()
            # before training.
            # ----------------------------------------------------------------
            shift_x=torch.tensor([4.3831, 4.9481, 4.2453, 5.1684, 4.1781, 4.3588]),
            scale_x=torch.tensor([0.6095, 0.6428, 0.594, 0.8557, 1.8731, 0.6863]),
            shift_z=torch.tensor([4.3866, 5.0113, 4.3238, 5.1864, 4.3652, 4.1874]),
            scale_z=torch.tensor([0.6086, 0.6285, 0.5973, 0.8442, 1.2282, 0.7623]),
        ),
    )

    observables_x: Tuple[Observable] = (
        Observable(
            name="lep_pt",
            compute=lambda x: x[..., 0],
            label=r"$p_{T,\ell}\ [\mathrm{GeV}]$",
            qlims=(1e-3, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="b1_pt",
            compute=lambda x: x[..., 1],
            label=r"$p_{T,b_1}\ [\mathrm{GeV}]$",
            qlims=(1e-3, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="b2_pt",
            compute=lambda x: x[..., 2],
            label=r"$p_{T,b_2}\ [\mathrm{GeV}]$",
            qlims=(1e-3, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="j1_pt",
            compute=lambda x: x[..., 3],
            label=r"$p_{T,j_1}\ [\mathrm{GeV}]$",
            qlims=(1e-3, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="j2_pt",
            compute=lambda x: x[..., 4],
            label=r"$p_{T,j_2}\ [\mathrm{GeV}]$",
            qlims=(1e-3, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="met_pt",
            compute=lambda x: x[..., 5],
            label=r"$p_{T,\mathrm{MET}}\ [\mathrm{GeV}]$",
            qlims=(1e-3, 1 - 1e-3),
            logy=True,
        ),
    )

    observables_z: Tuple[Observable] = (
        Observable(
            name="gen_lep_pt",
            compute=lambda z: z[..., 0],
            label=r"$p_{T,\ell}^{\mathrm{gen}}\ [\mathrm{GeV}]$",
            qlims=(1e-3, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="gen_b1_pt",
            compute=lambda z: z[..., 1],
            label=r"$p_{T,b_1}^{\mathrm{gen}}\ [\mathrm{GeV}]$",
            qlims=(1e-3, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="gen_b2_pt",
            compute=lambda z: z[..., 2],
            label=r"$p_{T,b_2}^{\mathrm{gen}}\ [\mathrm{GeV}]$",
            qlims=(1e-3, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="gen_j1_pt",
            compute=lambda z: z[..., 3],
            label=r"$p_{T,j_1}^{\mathrm{gen}}\ [\mathrm{GeV}]$",
            qlims=(1e-3, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="gen_j2_pt",
            compute=lambda z: z[..., 4],
            label=r"$p_{T,j_2}^{\mathrm{gen}}\ [\mathrm{GeV}]$",
            qlims=(1e-3, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="gen_nu_pt",
            compute=lambda z: z[..., 5],
            label=r"$p_{T,\nu}^{\mathrm{gen}}\ [\mathrm{GeV}]$",
            qlims=(1e-3, 1 - 1e-3),
            logy=True,
        ),
    )