from convert_to_parquet import convert_root_to_parquet

base = "/ptmp/mpp/mjosef/WbWb_analysis/FastFrames/output_ntuples_test"
#out_base = "/ptmp/mpp/mjosef/data_files/WbWb_files/Run3"
out_base = "/scratch/mjosef/Unfolding/aussie/data"

# ---- Sim: Pythia ttbar + DR singletop, to be reweighted/unfolded FROM ----
files_sim = [
    {"path": f"{base}/ttbar_601229_mc23d_fullsim.root", "dsid": 601229},
    {"path": f"{base}/ttbar_601230_mc23d_fullsim.root", "dsid": 601230},
    {"path": f"{base}/singletop_DR_601352_mc23d_fullsim.root", "dsid": 601352},
    {"path": f"{base}/singletop_DR_601355_mc23d_fullsim.root", "dsid": 601355},
]

# ---- Pseudo-data: Herwig ttbar, treated as target "Data" for the classifier ----
files_pseudodata = [
    {"path": f"{base}/ttbar_HW7_601414_mc23d_fullsim.root", "dsid": 601414},
    {"path": f"{base}/ttbar_HW7_601415_mc23d_fullsim.root", "dsid": 601415},
]

files_data = [
    {"path": f"{base}/data_0_2023_data.root", "dsid": 0},
]

if __name__ == "__main__":

    # quick sanity test first: just the smallest file
    #convert_root_to_parquet(
    #    [{"path": f"{base}/singletop_DR_601352_mc23d_fullsim.root", "dsid": 601352}],
    #    output_dir=f"{out_base}/sim_test",
    #    tree_name="reco",
    #    has_truth=True,
    #    step_size="1 GB",
    #)

    # once the test above succeeds, comment it out and uncomment these:

    # convert_root_to_parquet(
    #     files_sim,
    #     output_dir=f"{out_base}/sim",
    #     tree_name="reco",
    #     has_truth=True,
    #     step_size="1 GB",
    # )
    # convert_root_to_parquet(
    #     files_pseudodata,
    #     output_dir=f"{out_base}/pseudodata",
    #     tree_name="reco",
    #     has_truth=True,
    #     step_size="1 GB",
    # )
     convert_root_to_parquet(
         files_data,
         output_dir=f"{out_base}/data",
         tree_name="reco",
         has_truth=False,
         step_size="1 GB",
     )