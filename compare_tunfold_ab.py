"""
Compares four binned unfolding variants for the wwbb_multi pipeline:

  A       : column-normalized response matrix, unfolded via UNREGULARIZED
            pseudo-inverse (equivalent to compute_tunfold_result).
  B       : row-normalized response matrix, unfolded via DIRECT
            (one-shot) correction -- no inversion, automatically
            non-negative, but biased toward the sim truth prior
            (equivalent to the zeroth iteration of D'Agostini-style
            "Bayesian" unfolding).
  A*, B*  : same as above, but the SIM POPULATION used to build the
            response matrix is reweighted by a converged OmniFold/AUSSIE
            run's final per-event weight (pseudodata is UNCHANGED).
            Comparing A to A* (and B to B*) tests whether OmniFold
            handles regularization "for free" by correcting the prior --
            if A* is dramatically better-conditioned/less unstable than
            A, the original instability was largely a prior/migration-
            shape mismatch; if A* is STILL unstable, the ill-posedness
            is intrinsic to the detector's migrations for that
            observable.

Also produces heatmaps of A/A*/B/B* themselves, the raw joint histogram
M (log-scaled, to expose low-statistics bins that normalization alone
would hide), and an (A* - A) difference heatmap showing exactly where
OmniFold's reweighting redistributes probability mass in the response.

Usage:
    python compare_tunfold_ab.py \
        --omnifold_unf_dir runs/iterate_wwbb_multi/2026-07-24_13-51-45/it_20/unf \
        --num_bins_truth 15 --reco_bin_factor 2
"""

import argparse
import os
import sys
from contextlib import ExitStack

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import OmegaConf
from torch.utils.data import random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.metrics import compute_chi2, compute_wasserstein, compute_mse
from src.utils.tunfold import plot_correlation_matrix, plot_matrix_heatmap
from src.utils.tunfold_ab import compute_ab_results, load_omnifold_final_weight, matrices_from_joint

BENCHMARK_OBS = ["mu_pt", "mu_e", "j1_pt", "j1_e", "j2_pt", "j2_e", "j3_pt", "j3_e"]

COLOR_DATA = "#323232"
COLOR_SIM  = "#B22222"
COLOR_A      = "#1B2A4A"   # dark navy
COLOR_A_STAR = "#13B2B7"   # teal
COLOR_B      = "#7E3B91"   # purple
COLOR_B_STAR = "#FF8C00"   # orange

TAG_COLOR = {"A": COLOR_A, "A_star": COLOR_A_STAR, "B": COLOR_B, "B_star": COLOR_B_STAR}
TAG_LABEL = {"A": "A (pinv)", "A_star": "A* (pinv, OmniFold prior)",
             "B": "B (direct)", "B_star": "B* (direct, OmniFold prior)"}


def load_split(omnifold_unf_dir):
    cfg = OmegaConf.load(os.path.join(omnifold_unf_dir, ".hydra", "config.yaml"))
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
    labels = batch.labels
    mask_sim = (labels == 0).numpy()
    mask_dat = ~mask_sim
    return dset, process, batch, mask_sim, mask_dat


def get_observable(process, name, level):
    obs_list = process.observables_x if level == "x" else process.observables_z
    suffix = "" if level == "x" else "_truth"
    for obs in obs_list:
        if obs.name == f"{name}{suffix}":
            return obs
    raise KeyError(f"Observable '{name}{suffix}' not found")


def dup(a):
    return np.append(a, a[-1])


def main(omnifold_unf_dir, out_path="compare_tunfold_ab.pdf",
         num_bins_truth=15, reco_bin_factor=2, clip_negative=True):

    print(f"Loading test split + dataset from: {omnifold_unf_dir}")
    dset, process, batch, mask_sim, mask_dat = load_split(omnifold_unf_dir)

    x_all = batch.x if dset.aux_x is None else batch.aux_x
    z_all = batch.z if dset.aux_z is None else batch.aux_z
    x_sim_full, x_dat_full = x_all[mask_sim], x_all[mask_dat]
    z_sim_full = z_all[mask_sim]

    mask_z_all = batch.mask_z
    z_dat_check = batch.z[mask_dat]
    if mask_z_all is not None:
        data_has_truth = bool(mask_z_all[mask_dat].any())
    else:
        data_has_truth = bool((z_dat_check != 0).any())
    z_dat_full = z_all[mask_dat] if data_has_truth else None

    sample_logweights = batch.sample_logweights
    if sample_logweights is not None:
        lw_sample = sample_logweights.numpy()
        sim_w_raw = np.exp(lw_sample[mask_sim])
        exp_w = np.exp(lw_sample[mask_dat])
    else:
        sim_w_raw = np.ones(mask_sim.sum())
        exp_w = None

    print("Loading OmniFold final weight (sim-only, already includes base MC weight)")
    sim_w_star = load_omnifold_final_weight(omnifold_unf_dir)
    assert len(sim_w_star) == len(sim_w_raw), (
        f"OmniFold weight array length ({len(sim_w_star)}) does not match "
        f"sim test-set size ({len(sim_w_raw)}) -- dataset config mismatch?"
    )

    print(f"sim_w_raw : sum={sim_w_raw.sum():.1f}, N={len(sim_w_raw)}")
    print(f"sim_w_star: sum={sim_w_star.sum():.1f}, N={len(sim_w_star)}")
    if exp_w is not None:
        print(f"exp_w     : sum={exp_w.sum():.1f}, N={len(exp_w)}")

    metrics = {m: {tag: [] for tag in TAG_COLOR} for m in ("chi2", "wasserstein")}
    cond_table = []

    corr_path = out_path.replace(".pdf", "_correlations.pdf")
    matrices_path = out_path.replace(".pdf", "_matrices.pdf")

    with ExitStack() as stack:
        pdf = stack.enter_context(PdfPages(out_path))
        pdf_corr = stack.enter_context(PdfPages(corr_path))
        pdf_matrices = stack.enter_context(PdfPages(matrices_path))

        for name in BENCHMARK_OBS:
            obs_x = get_observable(process, name, "x")
            obs_z = get_observable(process, name, "z")

            x_sim_v = obs_x.compute(x_sim_full).numpy()
            x_dat_v = obs_x.compute(x_dat_full).numpy()
            z_sim_v = obs_z.compute(z_sim_full).numpy()
            z_dat_v = obs_z.compute(z_dat_full).numpy() if data_has_truth else None

            results, joint_histograms = compute_ab_results(
                x_sim_v, x_dat_v, z_sim_v, sim_w_raw, sim_w_star, exp_w,
                obs_x, obs_z, num_bins_truth, reco_bin_factor, clip_negative,
            )

            cond_table.append((
                name,
                results["A"].cond_number, results["A_star"].cond_number,
                results["A"].n_negative_bins, results["A_star"].n_negative_bins,
            ))

            reco_edges = results["A"].reco_edges
            truth_edges = results["A"].truth_edges

            # --------------------------------------------------------------
            # Page 1 (pdf): main overlay + ratio-to-data panel
            # --------------------------------------------------------------
            qlims = obs_z.qlims or (0.005, 0.995)
            lo, hi = np.quantile(z_sim_v, qlims)
            bins = (np.logspace(np.log10(max(lo, 1e-10)), np.log10(hi), 15)
                    if obs_z.log_bins else np.linspace(lo, hi, 15))

            y_sim, _ = np.histogram(z_sim_v, bins=bins, weights=sim_w_raw, density=True)
            y_dat = None
            if data_has_truth:
                y_dat, _ = np.histogram(z_dat_v, bins=bins, weights=exp_w, density=True)

            fig, (ax, axr) = plt.subplots(
                2, 1, figsize=(6, 6), sharex=True,
                gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05),
            )

            if y_dat is not None:
                ax.step(bins, dup(y_dat), where="post", color=COLOR_DATA, lw=1.2, label="Data")
            ax.step(bins, dup(y_sim), where="post", color=COLOR_SIM, lw=1.0, ls="--", label="Sim (raw)")

            for tag in ("A", "B", "A_star", "B_star"):
                w = results[tag].weight
                y, _ = np.histogram(z_sim_v, bins=bins, weights=w, density=True)
                ax.step(bins, dup(y), where="post", color=TAG_COLOR[tag], lw=1.6, label=TAG_LABEL[tag])

                if y_dat is not None:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = np.divide(y, y_dat, out=np.full_like(y, np.nan), where=y_dat > 0)
                    axr.step(bins, dup(ratio), where="post", color=TAG_COLOR[tag], lw=1.4)

            if y_dat is not None:
                axr.axhline(1.0, color=COLOR_DATA, lw=1.0)
            axr.set_ylim(0.9, 1.1)
            axr.set_ylabel("Pred. / Data", fontsize=9)
            axr.set_xlabel(obs_z.label)
            if obs_z.log_bins:
                ax.set_xscale("log")
                axr.set_xscale("log")

            ax.set_ylabel("Density")
            ax.legend(frameon=False, fontsize=7, ncol=2)
            ax.set_title(name, fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # --------------------------------------------------------------
            # Page 2 (pdf_corr): correlation matrices for A/B/A*/B*
            # --------------------------------------------------------------
            for tag in ("A", "B", "A_star", "B_star"):
                fig_c = plot_correlation_matrix(
                    results[tag].corr, results[tag].truth_edges,
                    xlabel=obs_z.label,
                    title=f"{TAG_LABEL[tag]} truth-bin correlation: {name}",
                )
                pdf_corr.savefig(fig_c)
                plt.close(fig_c)

            # --------------------------------------------------------------
            # Page 3 (pdf_matrices): response/joint-histogram heatmaps
            # --------------------------------------------------------------
            A_mat, B_mat = matrices_from_joint(joint_histograms["A"])
            A_star_mat, B_star_mat = matrices_from_joint(joint_histograms["A_star"])

            fig_m = plot_matrix_heatmap(
                A_mat, reco_edges, truth_edges, xlabel=obs_z.label,
                title=f"A: response matrix P(reco|truth) -- {name}",
                vmin=0, vmax=1,
            )
            pdf_matrices.savefig(fig_m)
            plt.close(fig_m)

            fig_m = plot_matrix_heatmap(
                A_star_mat, reco_edges, truth_edges, xlabel=obs_z.label,
                title=f"A*: response matrix P(reco|truth), OmniFold prior -- {name}",
                vmin=0, vmax=1,
            )
            pdf_matrices.savefig(fig_m)
            plt.close(fig_m)

            fig_m = plot_matrix_heatmap(
                B_mat.T, reco_edges, truth_edges, xlabel=obs_z.label,
                title=f"B: P(truth|reco), transposed for display -- {name}",
                vmin=0, vmax=1,
            )
            pdf_matrices.savefig(fig_m)
            plt.close(fig_m)

            fig_m = plot_matrix_heatmap(
                B_star_mat.T, reco_edges, truth_edges, xlabel=obs_z.label,
                title=f"B*: P(truth|reco), OmniFold prior, transposed -- {name}",
                vmin=0, vmax=1,
            )
            pdf_matrices.savefig(fig_m)
            plt.close(fig_m)

            fig_m = plot_matrix_heatmap(
                joint_histograms["A"], reco_edges, truth_edges, xlabel=obs_z.label,
                title=f"Raw joint histogram M (log10 counts), raw sim -- {name}",
                log_counts=True,
            )
            pdf_matrices.savefig(fig_m)
            plt.close(fig_m)

            fig_m = plot_matrix_heatmap(
                joint_histograms["A_star"], reco_edges, truth_edges, xlabel=obs_z.label,
                title=f"Raw joint histogram M (log10 counts), OmniFold-reweighted sim -- {name}",
                log_counts=True,
            )
            pdf_matrices.savefig(fig_m)
            plt.close(fig_m)

            fig_m = plot_matrix_heatmap(
                A_star_mat - A_mat, reco_edges, truth_edges, xlabel=obs_z.label,
                title=f"A* - A (OmniFold reweighting effect on response) -- {name}",
                diverging=True,
            )
            pdf_matrices.savefig(fig_m)
            plt.close(fig_m)

            # --------------------------------------------------------------
            # Metrics vs data truth
            # --------------------------------------------------------------
            if data_has_truth:
                for m_name, m_fn in (
                    ("chi2", compute_chi2),
                    ("wasserstein", compute_wasserstein),
                ):
                    kwargs = dict(num_bins=15, qlims=qlims) if m_name != "wasserstein" else {}
                    for tag in TAG_COLOR:
                        val = m_fn(z_dat_v, z_sim_v, exp_w, results[tag].weight, **kwargs)
                        metrics[m_name][tag].append(val)

        # ------------------------------------------------------------------
        # Metrics bar charts (pdf, appended after the per-observable pages)
        # ------------------------------------------------------------------
        if data_has_truth:
            for m_name, ylabel, hline, logy in (
                ("chi2", r"$\chi^2/N$", 1.0, True),
                ("wasserstein", "Wasserstein distance", None, False),
            ):
                fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(BENCHMARK_OBS)), 5))
                x = np.arange(len(BENCHMARK_OBS))
                width = 0.2
                for i, tag in enumerate(("A", "B", "A_star", "B_star")):
                    offset = (i - 1.5) * width
                    vals = np.nan_to_num(metrics[m_name][tag], nan=0.0)
                    ax.bar(x + offset, vals, width=width, label=TAG_LABEL[tag], color=TAG_COLOR[tag])
                if hline is not None:
                    ax.axhline(hline, color="black", linestyle="--", lw=1.2, label=f"Optimum ({hline:g})")
                ax.set_xticks(x)
                ax.set_xticklabels(BENCHMARK_OBS, rotation=30, ha="right", fontsize=9)
                ax.set_ylabel(ylabel, fontsize=10)
                if logy:
                    ax.set_yscale("log")
                ax.legend(frameon=False, fontsize=8)
                ax.set_title(f"Truth-level: {m_name} per observable (A/B/A*/B*)", fontsize=12)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    print(f"\nWrote {out_path}, {corr_path}, {matrices_path}\n")

    print(f"{'observable':<10} {'cond(A)':>12} {'cond(A*)':>12} {'#neg(A)':>9} {'#neg(A*)':>10}")
    print("-" * 56)
    for name, cA, cAs, nA, nAs in cond_table:
        print(f"{name:<10} {cA:>12.3g} {cAs:>12.3g} {nA:>9d} {nAs:>10d}")

    print(
        "\ncond(A) >> cond(A*): OmniFold's prior correction substantially "
        "improved the response matrix's conditioning -- the original A's "
        "instability was largely a prior/migration-shape mismatch.\n"
        "cond(A) ~ cond(A*): the ill-posedness is intrinsic to the detector's "
        "migrations for this observable -- persists even given a near-"
        "perfect prior; only genuine regularization or an unbinned "
        "approach (AUSSIE) can address it."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--omnifold_unf_dir", required=True,
                         help="Path to a converged OmniFold/AUSSIE 'unf' directory, "
                              "e.g. runs/iterate_wwbb_multi/<ts>/it_20/unf")
    parser.add_argument("--out", default="compare_tunfold_ab.pdf")
    parser.add_argument("--num_bins_truth", type=int, default=15)
    parser.add_argument("--reco_bin_factor", type=float, default=2)
    parser.add_argument("--no_clip", action="store_true")
    args = parser.parse_args()
    main(args.omnifold_unf_dir, args.out, args.num_bins_truth,
         args.reco_bin_factor, clip_negative=not args.no_clip)