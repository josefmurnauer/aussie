"""
Visualizes how the OmniFold statistical-uncertainty covariance/
correlation structure evolves across iterations, using predictions
already on disk (no retraining) -- for both the data-bootstrap-varying
run and, if provided, a matching control run.

Usage:
    python compare_uncertainty_evolution.py \
        --data_exp_dir runs/iterate_wwbb_multi/<ts> \
        --control_exp_dir runs/iterate_wwbb_multi_control/<ts_ctrl>
"""

import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import OmegaConf
from torch.utils.data import random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.uncertainty import (
    histogram_covariance_from_replicas,
    decompose_data_stat_covariance,
    plot_scalar_evolution,
    plot_correlation_matrix_grid,
)

BENCHMARK_OBS_Z = ["mu_pt_truth", "mu_e_truth", "j1_pt_truth", "j1_e_truth",
                   "j2_pt_truth", "j2_e_truth", "j3_pt_truth", "j3_e_truth"]

GRID_ITERATIONS = [1, 5, 10, 15, 20]  # representative iterations for the heatmap grid


def discover_iterations(exp_dir):
    dirs = glob.glob(os.path.join(exp_dir, "it_*"))

    def it_num(d):
        m = re.search(r"it_(\d+)", os.path.basename(d))
        return int(m.group(1)) if m else -1

    dirs = [d for d in dirs if it_num(d) >= 0]
    dirs.sort(key=it_num)
    return [(it_num(d), os.path.join(d, "unf")) for d in dirs]


def load_split_and_process(unf_dir):
    cfg = OmegaConf.load(os.path.join(unf_dir, ".hydra", "config.yaml"))
    dset = instantiate(cfg.dataset.reader)
    process = instantiate(cfg.dataset.process)

    fixed_rng = torch.Generator().manual_seed(1729)
    dcfg = cfg.data
    splits = random_split(
        dset,
        [1 - dcfg.val_frac - dcfg.test_frac, dcfg.val_frac, dcfg.test_frac],
        generator=fixed_rng,
    )
    batch = splits[2][:]
    mask_sim = (batch.labels == 0).numpy()
    z_sim = (batch.z if dset.aux_z is None else batch.aux_z)[mask_sim]
    return process, z_sim


def get_observable(process, name):
    for obs in process.observables_z:
        if obs.name == name:
            return obs
    raise KeyError(name)


def main(data_exp_dir, control_exp_dir=None, out_path="uncertainty_evolution.pdf"):

    data_its = discover_iterations(data_exp_dir)
    control_its = dict(discover_iterations(control_exp_dir)) if control_exp_dir else {}

    if not data_its:
        raise RuntimeError(f"No it_N/unf directories found under {data_exp_dir}")

    # process/binning taken from the first available iteration (dataset
    # config is identical across iterations)
    process, z_sim_ref = load_split_and_process(data_its[0][1])

    with PdfPages(out_path) as pdf:
        for obs_name in BENCHMARK_OBS_Z:
            obs = get_observable(process, obs_name)

            lo, hi = obs.xlims or tuple(np.quantile(z_sim_ref, obs.qlims or (0.005, 0.995)))
            if obs.log_bins:
                lo = max(lo, 1e-10)
                bins = np.logspace(np.log10(lo), np.log10(hi), 15)
            else:
                bins = np.linspace(lo, hi, 15)

            iters = []
            mean_rel_conflated = []
            mean_rel_clean = []
            n_neg_clean = []

            corr_grid_conflated = []
            corr_grid_clean = []
            grid_labels = []

            for it_num, unf_dir in data_its:
                pred_path = os.path.join(unf_dir, "predictions_test.npz")
                if not os.path.exists(pred_path):
                    continue

                _, z_sim = load_split_and_process(unf_dir)
                z_sim_v = obs.compute(z_sim).numpy()

                lw_z_sim = np.load(pred_path)["lw_z_sim"]
                if lw_z_sim.shape[0] <= 1:
                    print(f"it_{it_num}: no replica dimension found (K=1) -- skipping")
                    continue

                w_data = np.exp(lw_z_sim)
                mean_r, cov_r, corr_r = histogram_covariance_from_replicas(z_sim_v, w_data, bins)

                rel_err_conflated = np.sqrt(np.clip(np.diag(cov_r), 0, None))
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = np.divide(rel_err_conflated, mean_r,
                                     out=np.full_like(mean_r, np.nan), where=mean_r > 0)
                mean_rel_conflated.append(np.nanmean(rel))

                cov_clean = corr_clean = None
                if it_num in control_its:
                    control_unf_dir = control_its[it_num]
                    control_pred_path = os.path.join(control_unf_dir, "predictions_test.npz")
                    if os.path.exists(control_pred_path):
                        lw_z_sim_c = np.load(control_pred_path)["lw_z_sim"]
                        if lw_z_sim_c.shape[0] == lw_z_sim.shape[0]:
                            w_ctrl = np.exp(lw_z_sim_c)
                            mean_c, cov_c, _ = histogram_covariance_from_replicas(z_sim_v, w_ctrl, bins)
                            cov_clean, corr_clean, n_neg = decompose_data_stat_covariance(cov_r, cov_c)

                            rel_err_clean = np.sqrt(np.clip(np.diag(cov_clean), 0, None))
                            with np.errstate(divide="ignore", invalid="ignore"):
                                rel_c = np.divide(rel_err_clean, mean_r,
                                                   out=np.full_like(mean_r, np.nan), where=mean_r > 0)
                            mean_rel_clean.append(np.nanmean(rel_c))
                            n_neg_clean.append(n_neg)

                iters.append(it_num)

                if it_num in GRID_ITERATIONS:
                    corr_grid_conflated.append(corr_r)
                    if corr_clean is not None:
                        corr_grid_clean.append(corr_clean)
                    grid_labels.append(f"it_{it_num}")

            # ---- scalar evolution line plot ----
            series = {"Conflated (bootstrap-ensemble)": mean_rel_conflated}
            if len(mean_rel_clean) == len(iters):
                series["Clean (control-subtracted)"] = mean_rel_clean

            fig = plot_scalar_evolution(
                iters, series,
                ylabel="Mean relative statistical error (across bins)",
                title=f"AUSSIE data-stat evolution: {obs_name}",
            )
            pdf.savefig(fig)
            plt.close(fig)

            # ---- correlation matrix grid, conflated ----
            if corr_grid_conflated:
                fig = plot_correlation_matrix_grid(
                    corr_grid_conflated, grid_labels, bins, xlabel=obs.label,
                    title=f"Conflated bootstrap-ensemble correlation evolution: {obs_name}",
                )
                pdf.savefig(fig)
                plt.close(fig)

            # ---- correlation matrix grid, clean (control-subtracted) ----
            if corr_grid_clean:
                fig = plot_correlation_matrix_grid(
                    corr_grid_clean, grid_labels[:len(corr_grid_clean)], bins, xlabel=obs.label,
                    title=f"CLEAN (control-subtracted) correlation evolution: {obs_name}",
                )
                pdf.savefig(fig)
                plt.close(fig)

            print(f"{obs_name}: iterations processed = {iters}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_exp_dir", required=True)
    parser.add_argument("--control_exp_dir", default=None)
    parser.add_argument("--out", default="uncertainty_evolution.pdf")
    args = parser.parse_args()
    main(args.data_exp_dir, args.control_exp_dir, args.out)