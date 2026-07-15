import os
import numpy as np
import awkward as ak
import uproot


# ----------------------------------------------------------------------------
# Branch groups -- update here if any names turn out to differ
# ----------------------------------------------------------------------------

RECO_BRANCHES = [
    # muon
    "mu_pt_NOSYS", "mu_eta", "mu_phi", "mu_e_NOSYS",
    # met
    "met_met_NOSYS", "met_phi_NOSYS",
    # jets (jagged, full collection)
    "jet_pt_NOSYS", "jet_eta", "jet_phi", "jet_e_NOSYS",
    "jet_multiplicity_NOSYS",
    # jet tagging
    "jet_GN2v01", "jet_GN2v01_Continuous_quantile",
    "jet_GN2v01_pb", "jet_GN2v01_pc", "jet_GN2v01_pu", "jet_GN2v01_ptau",
    # leading b-jet (single object, reco) -- stored but not yet used by the network
    "jet1_bjet_pt_NOSYS", "jet1_bjet_eta_NOSYS",
    "jet1_bjet_phi_NOSYS", "jet1_bjet_e_NOSYS",
    # weights & identifiers
    "weight_total_NOSYS", "weight_total_squared_NOSYS",
    "eventNumber", "runNumber",
]

PARTICLE_BRANCHES = [
    # muon
    "particleLevel_PL_mu_pt", "particleLevel_PL_mu_eta",
    "particleLevel_PL_mu_phi", "particleLevel_PL_mu_e",
    # met
    "particleLevel_PL_met_met", "particleLevel_PL_met_phi",
    # jets (jagged)
    "particleLevel_PL_jet_pt", "particleLevel_PL_jet_eta",
    "particleLevel_PL_jet_phi", "particleLevel_PL_jet_e",
    "particleLevel_PL_jet_nGhosts_bHadron",
    # leading b-jet (single object, particle level)
    "PL_bjet1_pt_GEV_NOSYS", "PL_bjet1_eta_NOSYS",
    "PL_bjet1_phi_NOSYS", "PL_bjet1_e_NOSYS",
    # particle-level generator weight
    "particleLevel_PL_weight_mc_NOSYS",
]

# Human-readable process name per DSID -- used only for filenames/logging,
# NOT written into the parquet files (avoids the object-dtype issue with
# ak.with_field, and keeps files lean since dsid alone is enough provenance)
DSID_TO_PROCESS = {
    601229: "ttbar_pythia",
    601230: "ttbar_pythia",
    601352: "singletop",
    601355: "singletop",
    601414: "ttbar_herwig",
    601415: "ttbar_herwig",
}


def convert_root_to_parquet(
    files: list[dict],
    output_dir: str,
    tree_name: str = "reco",
    has_truth: bool = True,
    step_size: str = "200 MB",
):
    """
    Convert a list of ROOT files (all belonging to the same *channel*,
    e.g. all "sim" or all "pseudodata") into chunked Parquet files.

    Each input file is read in bounded-memory chunks via uproot.iterate
    and written out as one Parquet file per chunk, tagged with its
    source DSID for later bookkeeping.

    Parameters
    ----------
    files : list[dict]
        Each dict must have keys "path", "dsid", e.g.
        {"path": "/.../ttbar_601229_mc23d_fullsim.root", "dsid": 601229}
    output_dir : str
        Directory to write parquet chunks into (one directory per channel,
        e.g. .../data/sim/ or .../data/pseudodata/)
    tree_name : str
        Name of the TTree to read (confirmed "reco")
    has_truth : bool
        Whether to also read particle-level branches (True for both your
        Herwig pseudo-data and Pythia/singletop sim, since both are MC)
    step_size : str
        Chunk size passed to uproot.iterate, bounding memory usage
        regardless of total file size (e.g. safe even for the 22 GB file)
    """

    os.makedirs(output_dir, exist_ok=True)

    branches = list(RECO_BRANCHES)
    if has_truth:
        branches = branches + PARTICLE_BRANCHES
    branches = list(dict.fromkeys(branches))  # dedupe, preserve order

    chunk_idx_global = 0
    total_events = 0

    for finfo in files:
        path, dsid = finfo["path"], finfo["dsid"]
        process = DSID_TO_PROCESS.get(dsid, f"dsid{dsid}")
        print(f"Processing {path}  (dsid={dsid}, process={process})")

        chunk_idx = 0
        for chunk in uproot.iterate(
            f"{path}:{tree_name}",
            expressions=branches,
            step_size=step_size,
            library="ak",
        ):
            n = len(chunk)
            if n == 0:
                continue

            # attach provenance column (numeric only -- avoids object dtype
            # issues; process name can be looked up from dsid downstream
            # via DSID_TO_PROCESS)
            chunk = ak.with_field(chunk, np.full(n, dsid, dtype=np.int32), "dsid")

            out_path = os.path.join(
                output_dir, f"{process}_{dsid}_{chunk_idx:05d}.parquet"
            )
            ak.to_parquet(chunk, out_path)
            print(f"  wrote {n} events -> {out_path}")

            total_events += n
            chunk_idx += 1
            chunk_idx_global += 1

    print(
        f"Done: {chunk_idx_global} parquet files, "
        f"{total_events} total events -> {output_dir}"
    )