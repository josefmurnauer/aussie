import numpy as np
import os
import torch
import h5py

from collections import defaultdict
from dataclasses import dataclass
from tensordict import tensorclass
from typing import Callable, Tuple, Optional

from src.datasets.base_dataset import UnfoldingData
from src.utils.observable import Observable
from src.utils.transforms import LorentzTransform

# ---------------------------------------------------------------------------
# Particle ordering (fixed, 12 particles)
# ---------------------------------------------------------------------------
# Index  0  : lepton
# Index  1  : b-quark 1
# Index  2  : b-quark 2
# Index  3  : b-quark 3
# Index  4  : b-quark 4
# Index  5  : light jet 1
# Index  6  : light jet 2
# Index  7  : light jet 3
# Index  8  : light jet 4
# Index  9  : light jet 5
# Index  10 : light jet 6
# Index  11 : MET
# ---------------------------------------------------------------------------

N_PARTICLES = 12

# Particle type IDs (one-hot, 4 classes):
#   0 -> lepton
#   1 -> b-jet
#   2 -> light jet
#   3 -> MET
PARTICLE_ID_SEQ = [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3]

PARTICLE_NAMES_RECO = [
    r"\ell",
    r"b_1", r"b_2", r"b_3", r"b_4",
    r"j_1", r"j_2", r"j_3", r"j_4", r"j_5", r"j_6",
    r"\mathrm{MET}",
]

# Delta-R pairs of interest at reco level
DELTA_PAIRS_RECO = [
    (0, 1), (0, 2),              # lepton - b1, lepton - b2
    (1, 2), (1, 3), (2, 3),      # b-jet pairs
    (5, 6),                      # leading light jets
    (0, 11),                     # lepton - MET (W decay proxy)
    (1, 5), (2, 5),              # b-jet - leading light jet
]


# ---------------------------------------------------------------------------
# Coordinate conversion  (pt, eta, phi, mass) <-> (E, px, py, pz)
# ---------------------------------------------------------------------------


@tensorclass
class WWbbData(UnfoldingData):
    """
    Data class for WWbb -> lepton + jets decay.

    H5 file structure expected:
        file['reco']:    [#events, 12, #features]  - detector level (always present)
        file['gen']:     [#events, #particles, #features]  - particle level (OPTIONAL)
        file['weights']: [#events]                         - event weights

    4-vector convention per particle in the h5 file: (pt, eta, phi, mass)
        - pt   > 0  for real particles, == 0 for padding
        - eta  : pseudorapidity
        - phi  : azimuthal angle in [-pi, pi]
        - mass : invariant mass >= 0

    Internally the data is stored in (pt, eta, phi, mass) and converted to
    (E, px, py, pz) on-the-fly in the transform before passing to lGATr,
    which requires Cartesian 4-vectors.

    Particle ordering (reco, 12 fixed slots):
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

    --------------------------------
    Real (observed) data NEVER has a gen/particle level.
    The gen level is only available in MC simulation and is used
    for cross-checks / closure tests only.
    When gen is absent, 'z' and 'mask_z' are zero-padded so the
    tensorclass shape stays consistent.
    """

    @classmethod
    def read(
        cls,
        path_data:          str,
        path_mc:            str,
        device:             Optional[torch.device] = None,
        num:                Optional[int]          = None,
        max_particles_gen:  Optional[int]          = None,
    ):
        """
        Read WWbb data from h5 files.

        Args:
            path_data:          Path to the data h5 file (Herwig / observed).
                                May or may not contain a 'gen' dataset.
            path_mc:            Path to the MC h5 file (Pythia / simulated).
                                Expected to always contain a 'gen' dataset.
            device:             Torch device to place tensors on.
            num:                Optional max number of events to load.
            max_particles_gen:  Optional cap on gen-level particles.
                                Reco is always 12 particles (fixed).
        """

        # label=1 -> data (observed), label=0 -> mc (simulated)
        file_cfg = [
            ("data", path_data, 1),
            ("mc",   path_mc,   0),
        ]

        tensor_kwargs                            = defaultdict(list)
        batch_size                               = 0
        gen_shape_per_event: Optional[Tuple[int, int]] = None

        for split_name, path, label_val in file_cfg:
            with h5py.File(path, "r") as f:

                has_gen = "gen" in f

                if split_name == "data" and not has_gen:
                    print(
                        f"[{split_name}] No 'gen' dataset found in {path}.\n"
                        "  -> Expected for real (observed) data. "
                        "z / mask_z will be zero-padded."
                    )
                elif split_name == "mc" and not has_gen:
                    print(
                        f"[{split_name}] WARNING: No 'gen' dataset found in "
                        f"MC file {path}. Cross-checks will not be possible."
                    )

                # --- reco (always present, fixed 12 particles) -----------
                reco_raw    = f["reco"]   [:num].astype(np.float32)
                weights_raw = f["weights"][:num].astype(np.float32)

                assert reco_raw.shape[1] == N_PARTICLES, (
                    f"[{split_name}] Expected {N_PARTICLES} reco particles, "
                    f"got {reco_raw.shape[1]}. Check your h5 file."
                )

                assert reco_raw.shape[2] >= 4, (
                    f"[{split_name}] Expected at least 4 features (pt,eta,phi,mass) "
                    f"per particle, got {reco_raw.shape[2]}."
                )

                # --- gen (optional) --------------------------------------
                if has_gen:
                    gen_raw = f["gen"][:num].astype(np.float32)

                    if gen_shape_per_event is None:
                        n_particles_gen = (
                            min(gen_raw.shape[1], max_particles_gen)
                            if max_particles_gen is not None
                            else gen_raw.shape[1]
                        )
                        gen_shape_per_event = (n_particles_gen, gen_raw.shape[2])
                else:
                    gen_raw = None

                print(
                    f"[{split_name}] "
                    f"reco: {reco_raw.shape}  |  "
                    f"gen: {gen_raw.shape if gen_raw is not None else 'N/A (real data)'}  |  "
                    f"weights: {weights_raw.shape}"
                )

            # --- optional gen particle cap -------------------------------
            if gen_raw is not None and max_particles_gen is not None:
                gen_raw = gen_raw[:, :max_particles_gen, :]

            # --- torch tensors ------------------------------------------
            n_events = reco_raw.shape[0]

            # store raw (pt, eta, phi, mass) – conversion to (E,px,py,pz)
            # happens in WWbbTransform.forward()
            x = torch.from_numpy(reco_raw)     # [N, 12, >=4]
            w = torch.from_numpy(weights_raw)  # [N]

            # pt is index 0: real particle has pt > 0, padding has pt == 0
            mask_x = x[..., 0] > 0   # [N, 12]

            if gen_raw is not None:
                z      = torch.from_numpy(gen_raw)
                mask_z = z[..., 0] > 0
            else:
                z      = None
                mask_z = None

            labels = torch.full([n_events], float(label_val), dtype=torch.float32)

            tensor_kwargs["x"].append(x)
            tensor_kwargs["z"].append(z)
            tensor_kwargs["mask_x"].append(mask_x)
            tensor_kwargs["mask_z"].append(mask_z)
            tensor_kwargs["weights"].append(w)
            tensor_kwargs["labels"].append(labels)
            tensor_kwargs["has_gen"].append(
                torch.full([n_events], float(has_gen), dtype=torch.float32)
            )

            batch_size += n_events

        # --- fill missing gen with zeros --------------------------------
        tensor_kwargs = cls._fill_missing_gen(
            tensor_kwargs,
            gen_shape_per_event=gen_shape_per_event,
        )

        for k in list(tensor_kwargs.keys()):
            tensor_kwargs[k] = torch.cat(tensor_kwargs[k], dim=0)

        return cls(
            batch_size=[batch_size],
            device=device,
            **tensor_kwargs,
        )

    @staticmethod
    def _fill_missing_gen(
        tensor_kwargs:       defaultdict,
        gen_shape_per_event: Optional[Tuple[int, int]],
    ) -> defaultdict:
        """
        Replace None entries in 'z' / 'mask_z' with zero tensors of the
        correct shape so torch.cat() can proceed.
        """

        if gen_shape_per_event is None:
            print(
                "WARNING: No file contained a 'gen' dataset. "
                "Using fallback shape (1, 4) for the z placeholder."
            )
            gen_shape_per_event = (1, 4)

        n_part, n_feat = gen_shape_per_event

        patched_z:      list = []
        patched_mask_z: list = []

        for z_tensor, mask_z_tensor, x_tensor in zip(
            tensor_kwargs["z"],
            tensor_kwargs["mask_z"],
            tensor_kwargs["x"],
        ):
            if z_tensor is None:
                n = x_tensor.shape[0]
                patched_z.append(
                    torch.zeros(n, n_part, n_feat, dtype=torch.float32)
                )
                patched_mask_z.append(
                    torch.zeros(n, n_part, dtype=torch.bool)
                )
            else:
                patched_z.append(z_tensor)
                patched_mask_z.append(mask_z_tensor)

        tensor_kwargs["z"]      = patched_z
        tensor_kwargs["mask_z"] = patched_mask_z

        return tensor_kwargs
    
def ptetaphim_to_epxpypz(p: torch.Tensor) -> torch.Tensor:
    """
    Convert (pt, eta, phi, mass) -> (E, px, py, pz).

    Parameters
    ----------
    p : torch.Tensor
        Shape (..., 4) with p[..., 0]=pt, p[..., 1]=eta,
                             p[..., 2]=phi, p[..., 3]=mass

    Returns
    -------
    torch.Tensor
        Shape (..., 4) with Cartesian 4-momenta (E, px, py, pz).
        Padding particles (pt==0) remain all-zero.
    """
    pt   = p[..., 0]
    eta  = p[..., 1]
    phi  = p[..., 2]
    mass = p[..., 3]

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    E  = torch.sqrt(
        torch.clamp(
            (pt * torch.cosh(eta)) ** 2 + mass ** 2,
            min=0.0,
        )
    )

    return torch.stack([E, px, py, pz], dim=-1)


def epxpypz_to_ptetaphim(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert (E, px, py, pz) -> (pt, eta, phi, mass).

    Parameters
    ----------
    p : torch.Tensor
        Shape (..., 4) with p[..., 0]=E, p[..., 1]=px,
                             p[..., 2]=py, p[..., 3]=pz
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Shape (..., 4) with (pt, eta, phi, mass).
        Padding particles (E==0) remain all-zero.
    """
    E  = p[..., 0]
    px = p[..., 1]
    py = p[..., 2]
    pz = p[..., 3]

    pt   = torch.sqrt(torch.clamp(px ** 2 + py ** 2, min=0.0))
    phi  = torch.arctan2(py, px)
    p3   = torch.sqrt(torch.clamp(px ** 2 + py ** 2 + pz ** 2, min=0.0))
    eta  = torch.arctanh(
        torch.clamp(pz / p3.clamp(min=eps), min=-1 + eps, max=1 - eps)
    )
    mass = torch.sqrt(
        torch.clamp(E ** 2 - px ** 2 - py ** 2 - pz ** 2, min=0.0)
    )

    return torch.stack([pt, eta, phi, mass], dim=-1)

@tensorclass
class WWbbTransform:
    """
    Full preprocessing pipeline for WWbb data.

    Steps
    -----
    1. Convert (pt, eta, phi, mass) -> (E, px, py, pz)
       Required because LorentzTransform / lGATr work in Cartesian space.

    2. Normalise by a typical energy scale so values are O(1).

    3. Append one-hot particle-type IDs (4 classes):
           0 -> lepton
           1 -> b-jet
           2 -> light jet
           3 -> MET

    4. Apply LorentzTransform (adds beam spurions, optional rest-frame boost).

    Parameters
    ----------
    scale : float
        Global energy scale [GeV] used to normalise E, px, py, pz.
        MET mass and eta are 0 by construction so they are unaffected.
    """

    scale: torch.Tensor   # e.g. torch.tensor(200.0)

    def forward(self, batch):

        n       = len(batch)
        n_types = 4   # lepton, b-jet, light-jet, MET

        # ----------------------------------------------------------------
        # 1. (pt, eta, phi, mass) -> (E, px, py, pz)
        #    Only use the first 4 features; additional features (if any)
        #    are kept as extra scalars.
        # ----------------------------------------------------------------
        x_4vec   = ptetaphim_to_epxpypz(batch.x[..., :4])   # [N, 12, 4]
        x_extra  = batch.x[..., 4:]                          # [N, 12, F-4]

        z_4vec   = ptetaphim_to_epxpypz(batch.z[..., :4])   # [N, P_z, 4]
        z_extra  = batch.z[..., 4:]                          # [N, P_z, F-4]

        # ----------------------------------------------------------------
        # 2. Normalise by energy scale
        # ----------------------------------------------------------------
        x_4vec = x_4vec / self.scale.to(x_4vec.device)
        z_4vec = z_4vec / self.scale.to(z_4vec.device)

        # ----------------------------------------------------------------
        # 3. One-hot particle-type IDs
        # ----------------------------------------------------------------
        # --- reco (12 fixed slots) --------------------------------------
        ids_x = torch.eye(n_types, device=batch.device)[
            None, PARTICLE_ID_SEQ, :
        ].expand(n, -1, -1)                                  # [N, 12, 4]

        # --- gen (variable, MC only) ------------------------------------
        # Adjust gen_id_seq to match your MC truth particle ordering.
        # Example: lepton=0, neutrino->MET=3, b1=1, b2=1, q1=2, q2=2
        gen_id_seq = [0, 3, 1, 1, 2, 2]
        gen_id_seq = gen_id_seq[: batch.z.shape[1]]

        ids_z = torch.eye(n_types, device=batch.device)[
            None, gen_id_seq, :
        ].expand(n, -1, -1)

        # ----------------------------------------------------------------
        # 4. Concatenate scalars: [extra_features | one_hot_ids]
        #    These become the scalar channel fed to LorentzTransform.
        # ----------------------------------------------------------------
        scalars_x = torch.cat([x_extra, ids_x], dim=-1)   # [N, 12, F-4+4]
        scalars_z = torch.cat([z_extra, ids_z], dim=-1)

        # Combine 4-vectors + scalars into a single feature tensor
        # Layout expected by LorentzTransform: [..., :4] = 4-vector
        #                                      [..., 4:] = scalars
        x_full = torch.cat([x_4vec, scalars_x], dim=-1)   # [N, 12, 4+(F-4+4)]
        z_full = torch.cat([z_4vec, scalars_z], dim=-1)

        # ----------------------------------------------------------------
        # 5. LorentzTransform (beam spurions + optional boost)
        #    Operates on the Cartesian 4-vectors in [..., :4]
        # ----------------------------------------------------------------
        batch.x, batch.mask_x = LorentzTransform.forward(
            x_full, mask=batch.mask_x
        )
        batch.z, batch.mask_z = LorentzTransform.forward(
            z_full, mask=batch.mask_z
        )

        return batch

    def reverse(self, batch):
        """
        Reverse the transform to go back to (pt, eta, phi, mass).
        LorentzTransform.reverse is not implemented upstream, so we
        only undo the scale and coordinate conversion here.
        """
        raise NotImplementedError(
            "Full reverse not available because LorentzTransform.reverse "
            "is not implemented. Implement it in transforms.py first."
        )

# ---------------------------------------------------------------------------
# Process descriptor
# ---------------------------------------------------------------------------

@dataclass
class WWbbProcess:
    """
    Process descriptor for WWbb -> lepton + jets.

    Reco (x): 12 fixed slots
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

    Gen (z): MC truth only (optional, zero-padded for real data)

    num_features: 4 (E, px, py, pz after transform) + 4 (one-hot IDs) = 8
    """

    num_features: int = 8

    transforms: Tuple[Callable] = (
        WWbbTransform(
            scale=torch.tensor(200.0),
        ),
    )

    # -----------------------------------------------------------------------
    # Reco-level observables
    # Each particle has (pt, eta, phi, mass) at indices (0, 1, 2, 3)
    # nanify() sets padded particles (pt==0) to NaN automatically
    # -----------------------------------------------------------------------
    observables_x: Tuple[Observable] = (

        # -------------------------------------------------------------------
        # Lepton  (index 0)
        # -------------------------------------------------------------------
        Observable(
            name="lep_pt",
            compute=lambda x: nanify(x[..., 0, :], x[..., 0, 0]),
            label=r"$p_{T,\ell}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="lep_eta",
            compute=lambda x: nanify(x[..., 0, :], x[..., 0, 1]),
            label=r"$\eta_{\ell}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="lep_phi",
            compute=lambda x: nanify(x[..., 0, :], x[..., 0, 2]),
            label=r"$\phi_{\ell}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="lep_mass",
            compute=lambda x: nanify(x[..., 0, :], x[..., 0, 3]),
            label=r"$M_{\ell}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # -------------------------------------------------------------------
        # b-jet 1  (index 1)
        # -------------------------------------------------------------------
        Observable(
            name="b1_pt",
            compute=lambda x: nanify(x[..., 1, :], x[..., 1, 0]),
            label=r"$p_{T,b_1}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="b1_eta",
            compute=lambda x: nanify(x[..., 1, :], x[..., 1, 1]),
            label=r"$\eta_{b_1}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="b1_phi",
            compute=lambda x: nanify(x[..., 1, :], x[..., 1, 2]),
            label=r"$\phi_{b_1}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="b1_mass",
            compute=lambda x: nanify(x[..., 1, :], x[..., 1, 3]),
            label=r"$M_{b_1}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # -------------------------------------------------------------------
        # b-jet 2  (index 2)
        # -------------------------------------------------------------------
        Observable(
            name="b2_pt",
            compute=lambda x: nanify(x[..., 2, :], x[..., 2, 0]),
            label=r"$p_{T,b_2}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="b2_eta",
            compute=lambda x: nanify(x[..., 2, :], x[..., 2, 1]),
            label=r"$\eta_{b_2}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="b2_phi",
            compute=lambda x: nanify(x[..., 2, :], x[..., 2, 2]),
            label=r"$\phi_{b_2}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="b2_mass",
            compute=lambda x: nanify(x[..., 2, :], x[..., 2, 3]),
            label=r"$M_{b_2}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # -------------------------------------------------------------------
        # b-jet 3  (index 3)
        # -------------------------------------------------------------------
        Observable(
            name="b3_pt",
            compute=lambda x: nanify(x[..., 3, :], x[..., 3, 0]),
            label=r"$p_{T,b_3}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="b3_eta",
            compute=lambda x: nanify(x[..., 3, :], x[..., 3, 1]),
            label=r"$\eta_{b_3}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="b3_phi",
            compute=lambda x: nanify(x[..., 3, :], x[..., 3, 2]),
            label=r"$\phi_{b_3}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="b3_mass",
            compute=lambda x: nanify(x[..., 3, :], x[..., 3, 3]),
            label=r"$M_{b_3}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # -------------------------------------------------------------------
        # b-jet 4  (index 4)
        # -------------------------------------------------------------------
        Observable(
            name="b4_pt",
            compute=lambda x: nanify(x[..., 4, :], x[..., 4, 0]),
            label=r"$p_{T,b_4}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="b4_eta",
            compute=lambda x: nanify(x[..., 4, :], x[..., 4, 1]),
            label=r"$\eta_{b_4}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="b4_phi",
            compute=lambda x: nanify(x[..., 4, :], x[..., 4, 2]),
            label=r"$\phi_{b_4}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="b4_mass",
            compute=lambda x: nanify(x[..., 4, :], x[..., 4, 3]),
            label=r"$M_{b_4}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # -------------------------------------------------------------------
        # Light jet 1  (index 5)
        # -------------------------------------------------------------------
        Observable(
            name="j1_pt",
            compute=lambda x: nanify(x[..., 5, :], x[..., 5, 0]),
            label=r"$p_{T,j_1}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="j1_eta",
            compute=lambda x: nanify(x[..., 5, :], x[..., 5, 1]),
            label=r"$\eta_{j_1}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="j1_phi",
            compute=lambda x: nanify(x[..., 5, :], x[..., 5, 2]),
            label=r"$\phi_{j_1}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="j1_mass",
            compute=lambda x: nanify(x[..., 5, :], x[..., 5, 3]),
            label=r"$M_{j_1}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # -------------------------------------------------------------------
        # Light jet 2  (index 6)
        # -------------------------------------------------------------------
        Observable(
            name="j2_pt",
            compute=lambda x: nanify(x[..., 6, :], x[..., 6, 0]),
            label=r"$p_{T,j_2}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="j2_eta",
            compute=lambda x: nanify(x[..., 6, :], x[..., 6, 1]),
            label=r"$\eta_{j_2}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="j2_phi",
            compute=lambda x: nanify(x[..., 6, :], x[..., 6, 2]),
            label=r"$\phi_{j_2}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="j2_mass",
            compute=lambda x: nanify(x[..., 6, :], x[..., 6, 3]),
            label=r"$M_{j_2}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # -------------------------------------------------------------------
        # Light jet 3  (index 7)
        # -------------------------------------------------------------------
        Observable(
            name="j3_pt",
            compute=lambda x: nanify(x[..., 7, :], x[..., 7, 0]),
            label=r"$p_{T,j_3}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="j3_eta",
            compute=lambda x: nanify(x[..., 7, :], x[..., 7, 1]),
            label=r"$\eta_{j_3}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="j3_phi",
            compute=lambda x: nanify(x[..., 7, :], x[..., 7, 2]),
            label=r"$\phi_{j_3}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="j3_mass",
            compute=lambda x: nanify(x[..., 7, :], x[..., 7, 3]),
            label=r"$M_{j_3}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # -------------------------------------------------------------------
        # Light jet 4  (index 8)
        # -------------------------------------------------------------------
        Observable(
            name="j4_pt",
            compute=lambda x: nanify(x[..., 8, :], x[..., 8, 0]),
            label=r"$p_{T,j_4}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="j4_eta",
            compute=lambda x: nanify(x[..., 8, :], x[..., 8, 1]),
            label=r"$\eta_{j_4}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="j4_phi",
            compute=lambda x: nanify(x[..., 8, :], x[..., 8, 2]),
            label=r"$\phi_{j_4}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="j4_mass",
            compute=lambda x: nanify(x[..., 8, :], x[..., 8, 3]),
            label=r"$M_{j_4}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # -------------------------------------------------------------------
        # Light jet 5  (index 9)
        # -------------------------------------------------------------------
        Observable(
            name="j5_pt",
            compute=lambda x: nanify(x[..., 9, :], x[..., 9, 0]),
            label=r"$p_{T,j_5}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="j5_eta",
            compute=lambda x: nanify(x[..., 9, :], x[..., 9, 1]),
            label=r"$\eta_{j_5}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="j5_phi",
            compute=lambda x: nanify(x[..., 9, :], x[..., 9, 2]),
            label=r"$\phi_{j_5}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="j5_mass",
            compute=lambda x: nanify(x[..., 9, :], x[..., 9, 3]),
            label=r"$M_{j_5}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # -------------------------------------------------------------------
        # Light jet 6  (index 10)
        # -------------------------------------------------------------------
        Observable(
            name="j6_pt",
            compute=lambda x: nanify(x[..., 10, :], x[..., 10, 0]),
            label=r"$p_{T,j_6}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="j6_eta",
            compute=lambda x: nanify(x[..., 10, :], x[..., 10, 1]),
            label=r"$\eta_{j_6}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="j6_phi",
            compute=lambda x: nanify(x[..., 10, :], x[..., 10, 2]),
            label=r"$\phi_{j_6}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="j6_mass",
            compute=lambda x: nanify(x[..., 10, :], x[..., 10, 3]),
            label=r"$M_{j_6}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # -------------------------------------------------------------------
        # MET  (index 11)
        # eta and mass are 0 by construction for MET
        # -------------------------------------------------------------------
        Observable(
            name="met_pt",
            compute=lambda x: nanify(x[..., 11, :], x[..., 11, 0]),
            label=r"$p_{T,\mathrm{MET}}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="met_phi",
            compute=lambda x: nanify(x[..., 11, :], x[..., 11, 2]),
            label=r"$\phi_{\mathrm{MET}}$",
            xlims=(-np.pi, np.pi),
        ),

        # -------------------------------------------------------------------
        # Delta-R pairs
        # -------------------------------------------------------------------
        Observable(
            name="dR_lep_b1",
            compute=lambda x: compute_deltaR(x, 0, 1),
            label=r"$\Delta R_{\ell, b_1}$",
            qlims=(0, 1 - 1e-3),
        ),
        Observable(
            name="dR_lep_b2",
            compute=lambda x: compute_deltaR(x, 0, 2),
            label=r"$\Delta R_{\ell, b_2}$",
            qlims=(0, 1 - 1e-3),
        ),
        Observable(
            name="dR_b1_b2",
            compute=lambda x: compute_deltaR(x, 1, 2),
            label=r"$\Delta R_{b_1, b_2}$",
            qlims=(0, 1 - 1e-3),
        ),
        Observable(
            name="dR_b1_b3",
            compute=lambda x: compute_deltaR(x, 1, 3),
            label=r"$\Delta R_{b_1, b_3}$",
            qlims=(0, 1 - 1e-3),
        ),
        Observable(
            name="dR_b2_b3",
            compute=lambda x: compute_deltaR(x, 2, 3),
            label=r"$\Delta R_{b_2, b_3}$",
            qlims=(0, 1 - 1e-3),
        ),
        Observable(
            name="dR_j1_j2",
            compute=lambda x: compute_deltaR(x, 5, 6),
            label=r"$\Delta R_{j_1, j_2}$",
            qlims=(0, 1 - 1e-3),
        ),
        Observable(
            name="dR_lep_met",
            compute=lambda x: compute_deltaR(x, 0, 11),
            label=r"$\Delta R_{\ell, \mathrm{MET}}$",
            qlims=(0, 1 - 1e-3),
        ),
        Observable(
            name="dR_b1_j1",
            compute=lambda x: compute_deltaR(x, 1, 5),
            label=r"$\Delta R_{b_1, j_1}$",
            qlims=(0, 1 - 1e-3),
        ),

        # -------------------------------------------------------------------
        # Invariant masses of physically meaningful combinations
        # -------------------------------------------------------------------
        Observable(
            name="m_lep_met",
            compute=lambda x: compute_invariant_mass(x, [0, 11]),
            label=r"$M_{\ell, \mathrm{MET}}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),
        Observable(
            name="m_b1_b2",
            compute=lambda x: compute_invariant_mass(x, [1, 2]),
            label=r"$M_{b_1, b_2}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),
        Observable(
            name="m_lep_b1_b2",
            compute=lambda x: compute_invariant_mass(x, [0, 1, 2]),
            label=r"$M_{\ell, b_1, b_2}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),
        Observable(
            name="m_j1_j2",
            compute=lambda x: compute_invariant_mass(x, [5, 6]),
            label=r"$M_{j_1, j_2}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),
    )

    # -----------------------------------------------------------------------
    # Gen-level observables  (MC only, zero-padded for real data)
    # Same (pt, eta, phi, mass) convention
    # Adjust names/indices to match your MC truth record
    # -----------------------------------------------------------------------
    observables_z: Tuple[Observable] = (

        # lepton  (index 0)
        Observable(
            name="gen_lep_pt",
            compute=lambda z: nanify(z[..., 0, :], z[..., 0, 0]),
            label=r"$p_{T,\ell}^{\mathrm{gen}}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="gen_lep_eta",
            compute=lambda z: nanify(z[..., 0, :], z[..., 0, 1]),
            label=r"$\eta_{\ell}^{\mathrm{gen}}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="gen_lep_phi",
            compute=lambda z: nanify(z[..., 0, :], z[..., 0, 2]),
            label=r"$\phi_{\ell}^{\mathrm{gen}}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="gen_lep_mass",
            compute=lambda z: nanify(z[..., 0, :], z[..., 0, 3]),
            label=r"$M_{\ell}^{\mathrm{gen}}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # neutrino  (index 1)
        Observable(
            name="gen_nu_pt",
            compute=lambda z: nanify(z[..., 1, :], z[..., 1, 0]),
            label=r"$p_{T,\nu}^{\mathrm{gen}}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="gen_nu_eta",
            compute=lambda z: nanify(z[..., 1, :], z[..., 1, 1]),
            label=r"$\eta_{\nu}^{\mathrm{gen}}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="gen_nu_phi",
            compute=lambda z: nanify(z[..., 1, :], z[..., 1, 2]),
            label=r"$\phi_{\nu}^{\mathrm{gen}}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="gen_nu_mass",
            compute=lambda z: nanify(z[..., 1, :], z[..., 1, 3]),
            label=r"$M_{\nu}^{\mathrm{gen}}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # b quark 1  (index 2)
        Observable(
            name="gen_b1_pt",
            compute=lambda z: nanify(z[..., 2, :], z[..., 2, 0]),
            label=r"$p_{T,b_1}^{\mathrm{gen}}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="gen_b1_eta",
            compute=lambda z: nanify(z[..., 2, :], z[..., 2, 1]),
            label=r"$\eta_{b_1}^{\mathrm{gen}}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="gen_b1_phi",
            compute=lambda z: nanify(z[..., 2, :], z[..., 2, 2]),
            label=r"$\phi_{b_1}^{\mathrm{gen}}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="gen_b1_mass",
            compute=lambda z: nanify(z[..., 2, :], z[..., 2, 3]),
            label=r"$M_{b_1}^{\mathrm{gen}}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # b quark 2  (index 3)
        Observable(
            name="gen_b2_pt",
            compute=lambda z: nanify(z[..., 3, :], z[..., 3, 0]),
            label=r"$p_{T,b_2}^{\mathrm{gen}}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="gen_b2_eta",
            compute=lambda z: nanify(z[..., 3, :], z[..., 3, 1]),
            label=r"$\eta_{b_2}^{\mathrm{gen}}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="gen_b2_phi",
            compute=lambda z: nanify(z[..., 3, :], z[..., 3, 2]),
            label=r"$\phi_{b_2}^{\mathrm{gen}}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="gen_b2_mass",
            compute=lambda z: nanify(z[..., 3, :], z[..., 3, 3]),
            label=r"$M_{b_2}^{\mathrm{gen}}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # light quark 1  (index 4)
        Observable(
            name="gen_q1_pt",
            compute=lambda z: nanify(z[..., 4, :], z[..., 4, 0]),
            label=r"$p_{T,q_1}^{\mathrm{gen}}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="gen_q1_eta",
            compute=lambda z: nanify(z[..., 4, :], z[..., 4, 1]),
            label=r"$\eta_{q_1}^{\mathrm{gen}}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="gen_q1_phi",
            compute=lambda z: nanify(z[..., 4, :], z[..., 4, 2]),
            label=r"$\phi_{q_1}^{\mathrm{gen}}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="gen_q1_mass",
            compute=lambda z: nanify(z[..., 4, :], z[..., 4, 3]),
            label=r"$M_{q_1}^{\mathrm{gen}}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # light quark 2  (index 5)
        Observable(
            name="gen_q2_pt",
            compute=lambda z: nanify(z[..., 5, :], z[..., 5, 0]),
            label=r"$p_{T,q_2}^{\mathrm{gen}}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
            logy=True,
        ),
        Observable(
            name="gen_q2_eta",
            compute=lambda z: nanify(z[..., 5, :], z[..., 5, 1]),
            label=r"$\eta_{q_2}^{\mathrm{gen}}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="gen_q2_phi",
            compute=lambda z: nanify(z[..., 5, :], z[..., 5, 2]),
            label=r"$\phi_{q_2}^{\mathrm{gen}}$",
            xlims=(-np.pi, np.pi),
        ),
        Observable(
            name="gen_q2_mass",
            compute=lambda z: nanify(z[..., 5, :], z[..., 5, 3]),
            label=r"$M_{q_2}^{\mathrm{gen}}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),

        # -------------------------------------------------------------------
        # Gen-level Delta-R and invariant mass combinations
        # -------------------------------------------------------------------
        Observable(
            name="gen_dR_lep_nu",
            compute=lambda z: compute_deltaR(z, 0, 1),
            label=r"$\Delta R_{\ell,\nu}^{\mathrm{gen}}$",
            qlims=(0, 1 - 1e-3),
        ),
        Observable(
            name="gen_dR_b1_b2",
            compute=lambda z: compute_deltaR(z, 2, 3),
            label=r"$\Delta R_{b_1,b_2}^{\mathrm{gen}}$",
            qlims=(0, 1 - 1e-3),
        ),
        Observable(
            name="gen_dR_q1_q2",
            compute=lambda z: compute_deltaR(z, 4, 5),
            label=r"$\Delta R_{q_1,q_2}^{\mathrm{gen}}$",
            qlims=(0, 1 - 1e-3),
        ),
        Observable(
            name="gen_dR_lep_b1",
            compute=lambda z: compute_deltaR(z, 0, 2),
            label=r"$\Delta R_{\ell,b_1}^{\mathrm{gen}}$",
            qlims=(0, 1 - 1e-3),
        ),
        Observable(
            name="gen_m_lep_nu",
            compute=lambda z: compute_invariant_mass(z, [0, 1]),
            label=r"$M_{\ell,\nu}^{\mathrm{gen}}$ (W mass proxy)",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),
        Observable(
            name="gen_m_b1_b2",
            compute=lambda z: compute_invariant_mass(z, [2, 3]),
            label=r"$M_{b_1,b_2}^{\mathrm{gen}}$",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),
        Observable(
            name="gen_m_q1_q2",
            compute=lambda z: compute_invariant_mass(z, [4, 5]),
            label=r"$M_{q_1,q_2}^{\mathrm{gen}}$ (W mass proxy)",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),
        Observable(
            name="gen_m_all",
            compute=lambda z: compute_invariant_mass(z, [0, 1, 2, 3, 4, 5]),
            label=r"$M_{\mathrm{all}}^{\mathrm{gen}}$ (ttbar proxy)",
            unit="GeV",
            qlims=(1e-4, 1 - 1e-3),
        ),
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def nanify(p: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Set observable to NaN for padded particles (pt == 0)."""
    return torch.where(
        p[..., 0] != 0.0,
        obs,
        torch.tensor(float("nan"), device=p.device, dtype=p.dtype),
    )


def compute_deltaR(p: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """
    DeltaR between particles at index i and j.
    p: [N, n_particles, 4]  with (pt, eta, phi, mass)
    """
    dphi = (p[..., i, 2] - p[..., j, 2] + np.pi) % (2 * np.pi) - np.pi
    deta =  p[..., i, 1] - p[..., j, 1]
    return nanify(
        p[..., i, :],
        torch.sqrt(dphi ** 2 + deta ** 2),
    )


def compute_invariant_mass(p: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """
    Invariant mass of the system formed by particles at given indices.
    Useful for W-boson or top-quark reconstruction checks.
    p: [N, n_particles, 4]  with (pt, eta, phi, mass)
    """
    cart = ptetaphim_to_epxpypz(p[:, indices, :])   # [N, k, 4]

    E_sum  = cart[..., 0].sum(-1)
    px_sum = cart[..., 1].sum(-1)
    py_sum = cart[..., 2].sum(-1)
    pz_sum = cart[..., 3].sum(-1)

    return torch.sqrt(
        torch.clamp(
            E_sum ** 2 - px_sum ** 2 - py_sum ** 2 - pz_sum ** 2,
            min=0.0,
        )
    )