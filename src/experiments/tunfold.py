import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from hydra.utils import instantiate
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import random_split

from src.experiments.base_experiment import BaseExperiment
from src.utils.metrics import (
    compute_chi2,
    compute_wasserstein,
    compute_mse,
    save_metrics_pdf,
    _flatten_metrics_dict,
)
from src.utils.tunfold import compute_tunfold_result, plot_correlation_matrix


class TUnfoldExperiment(BaseExperiment):
    """
    Binned baseline: for each benchmark observable, builds a 1D
    truth-reco migration matrix from simulation, then unfolds the
    observed (data/pseudodata) reco-level histogram via the UNREGULARIZED
    Moore-Penrose pseudo-inverse (no Tikhonov term, no SVD truncation --
    this is deliberately the "naive TUnfold" limit, cf. Schmitt, TUnfold,
    JINST 2012, with tau=0).

    Reco-level statistical uncertainties are assumed UNCORRELATED across
    reco bins (diagonal V_y); the resulting truth-level covariance
    V_x = A+ . V_y . (A+)^T is generally NOT diagonal -- any correlations
    appearing in the truth-bin covariance/correlation matrices are
    induced purely by the (unregularized) matrix inversion mixing reco
    bins together. This is exactly the instability this baseline is
    meant to illustrate, relative to OmniFold (IterationExperiment) and
    AUSSIE (UnfoldingExperiment).

    Produces:
        - plots/latents.pdf : Sim / Data(-truth) / TUnfold curves per
          observable (truth level only -- TUnfold has no meaningful
          reco-level output beyond the data histogram itself), with a
          correctly error-propagated uncertainty band on the TUnfold
          curve (not the generic per-event-weight formula used for
          Classifier/AUSSIE).
        - plots/tunfold_correlations.pdf : one truth-bin correlation
          matrix heatmap per observable, showing the correlation
          structure induced by the unregularized inversion.
        - plots/metrics.pdf, plots/metrics.npz : chi2/Wasserstein/MSE
          for the "TUnfold" curve under observables_z, in the exact same
          flattened format used by Classification-/UnfoldingExperiment,
          so downstream tooling (e.g. compare_benchmarks.py) can consume
          it with no changes beyond adding "TUnfold" as a curve name.

    Runs unconditionally on `run()` -- ignores cfg.train/evaluate/plot,
    since there is no model to train or evaluate.
    """

    def __init__(self, cfg, exp_dir):
        super().__init__(cfg, exp_dir)
        self.log = logging.getLogger("TUnfoldExperiment")

    def run(self):

        self.process = instantiate(self.cfg.dataset.process)

        if self.cfg.atlas_style:
            from src.utils import plotting_atlasstyle as plotting
        else:
            from src.utils import plotting

        pcfg = self.cfg.plotting
        pw = pcfg.pagewidth
        tcfg = self.cfg.tunfold

        savedir = os.path.join(self.exp_dir, "plots")
        os.makedirs(savedir, exist_ok=True)

        # ------------------------------------------------------------------
        # Load data using the SAME seeded split as Classification-/
        # UnfoldingExperiment, so results are directly comparable event-
        # for-event (given identical dataset config: num_sim/num_data/seed).
        # ------------------------------------------------------------------
        self.log.info("Loading dataset")
        dset = instantiate(self.cfg.dataset.reader)
        _, _, test_set = self.split_dataset(dset)
        batch = test_set[:]

        labels = batch.labels
        mask_sim = (labels == 0).numpy()
        mask_dat = ~mask_sim

        x_all = batch.x if dset.aux_x is None else batch.aux_x
        x_sim_full = x_all[mask_sim]
        x_dat_full = x_all[mask_dat]

        mask_z_all = batch.mask_z
        z_dat_check = batch.z[mask_dat]
        if mask_z_all is not None:
            data_has_truth = bool(mask_z_all[mask_dat].any())
        else:
            data_has_truth = bool((z_dat_check != 0).any())

        if not data_has_truth:
            self.log.info(
                "Data population has no particle-level truth information -- "
                "latents.pdf will show only the (unweighted vs. TUnfold-"
                "unfolded) sim truth curve, without a data-truth reference, "
                "and no chi2/Wasserstein/MSE metrics will be computed."
            )

        z_all = batch.z if dset.aux_z is None else batch.aux_z
        z_sim_full = z_all[mask_sim]
        z_dat_full = z_all[mask_dat] if data_has_truth else None

        sample_logweights = batch.sample_logweights
        if sample_logweights is not None:
            lw_sample = sample_logweights.numpy()
            sim_weights = np.exp(lw_sample[mask_sim])
            exp_weights = np.exp(lw_sample[mask_dat])
        else:
            sim_weights = None
            exp_weights = None

        num_bins_truth = tcfg.num_bins_truth
        reco_bin_factor = tcfg.reco_bin_factor
        clip_negative = tcfg.clip_negative

        self.log.info(
            f"TUnfold config: num_bins_truth={num_bins_truth}, "
            f"reco_bin_factor={reco_bin_factor} "
            f"(num_bins_reco={int(round(num_bins_truth * reco_bin_factor))}), "
            f"clip_negative={clip_negative}"
        )

        obs_names_z = []
        metrics_z = {
            "chi2": {"TUnfold": []},
            "wasserstein": {"TUnfold": []},
            "mse": {"TUnfold": []},
        }

        # ------------------------------------------------------------------
        # Per-observable response matrix + unregularized pseudo-inverse,
        # with full covariance/correlation propagation
        # ------------------------------------------------------------------
        with PdfPages(os.path.join(savedir, "latents.pdf")) as pdf, \
             PdfPages(os.path.join(savedir, "tunfold_correlations.pdf")) as pdf_corr:

            for obs_x, obs_z in zip(self.process.observables_x, self.process.observables_z):

                x_sim_v = obs_x.compute(x_sim_full).numpy()
                x_dat_v = obs_x.compute(x_dat_full).numpy()
                z_sim_v = obs_z.compute(z_sim_full).numpy()
                z_dat_v = obs_z.compute(z_dat_full).numpy() if data_has_truth else None

                result = compute_tunfold_result(
                    x_sim_v=x_sim_v,
                    x_dat_v=x_dat_v,
                    z_sim_v=z_sim_v,
                    z_dat_v=z_dat_v,
                    sim_weights=sim_weights,
                    exp_weights=exp_weights,
                    obs_x=obs_x,
                    obs_z=obs_z,
                    num_bins_truth=num_bins_truth,
                    reco_bin_factor=reco_bin_factor,
                    clip_negative=clip_negative,
                )
                weight = result.weight

                # ---- correlation matrix diagnostic page ----
                fig_corr = plot_correlation_matrix(
                    result.corr, result.truth_edges,
                    xlabel=obs_z.label,
                    title=f"TUnfold truth-bin correlation: {obs_z.name}",
                )
                pdf_corr.savefig(fig_corr)
                plt.close(fig_corr)

                obs_names_z.append(obs_z.name)

                if data_has_truth:
                    fig, ax = plotting.plot_reweighting(
                        exp=z_dat_v,
                        sim=z_sim_v,
                        weights_list=[weight],
                        variance_list=[result.logvar],
                        names_list=["TUnfold"],
                        xlabel=obs_z.label,
                        figsize=np.array([1, 7 / 8]) * pw / 2,
                        num_bins=pcfg.num_bins,
                        discrete=obs_z.discrete,
                        log_bins=obs_z.log_bins,
                        logy=obs_z.logy,
                        qlims=obs_z.qlims,
                        xlims=obs_z.xlims,
                        exp_weights=exp_weights,
                        sim_weights=sim_weights,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

                    if obs_z.log_bins:
                        fig, ax = plotting.plot_reweighting(
                            exp=z_dat_v,
                            sim=z_sim_v,
                            weights_list=[weight],
                            variance_list=[result.logvar],
                            names_list=["TUnfold"],
                            xlabel=obs_z.label,
                            figsize=np.array([1, 7 / 8]) * pw / 2,
                            num_bins=pcfg.num_bins,
                            discrete=False,
                            log_bins=True,
                            logx=True,
                            logy=obs_z.logy,
                            qlims=obs_z.qlims,
                            xlims=None,
                            exp_weights=exp_weights,
                            sim_weights=sim_weights,
                        )
                        pdf.savefig(fig)
                        plt.close(fig)

                    metrics_z["chi2"]["TUnfold"].append(
                        compute_chi2(
                            z_dat_v, z_sim_v, exp_weights, weight,
                            num_bins=pcfg.num_bins, qlims=obs_z.qlims, xlims=obs_z.xlims,
                        )
                    )
                    metrics_z["wasserstein"]["TUnfold"].append(
                        compute_wasserstein(z_dat_v, z_sim_v, exp_weights, weight)
                    )
                    metrics_z["mse"]["TUnfold"].append(
                        compute_mse(
                            z_dat_v, z_sim_v, exp_weights, weight,
                            num_bins=pcfg.num_bins, qlims=obs_z.qlims, xlims=obs_z.xlims,
                        )
                    )

                else:
                    # no data truth: show sim-only (unweighted baseline
                    # relabeled as "exp") vs. TUnfold-unfolded curve
                    fig, ax = plotting.plot_reweighting(
                        exp=z_sim_v,
                        sim=z_sim_v,
                        weights_list=[weight],
                        variance_list=[result.logvar],
                        names_list=["TUnfold"],
                        xlabel=obs_z.label,
                        figsize=np.array([1, 7 / 8]) * pw / 2,
                        num_bins=pcfg.num_bins,
                        discrete=obs_z.discrete,
                        log_bins=obs_z.log_bins,
                        logy=obs_z.logy,
                        qlims=obs_z.qlims,
                        xlims=obs_z.xlims,
                        exp_weights=None,
                        sim_weights=sim_weights,
                        name_exp="Sim (unweighted)",
                        show_sim=False,
                        add_chi2=False,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

        # ------------------------------------------------------------------
        # Save metrics in the same flattened format as Classification-/
        # UnfoldingExperiment, so compare_benchmarks.py-style tooling can
        # load it with `metrics.npz["observables_z/chi2/TUnfold"]` etc.
        # ------------------------------------------------------------------
        metrics_to_save = {}
        if data_has_truth:
            with PdfPages(os.path.join(savedir, "metrics.pdf")) as pdf:
                save_metrics_pdf(pdf, obs_names_z, metrics_z, prefix="Truth-level: ")
            metrics_to_save["observables_z"] = {"names": obs_names_z, **metrics_z}
        else:
            self.log.info("Skipping metrics.pdf/npz -- no data-truth available")

        np.savez(
            os.path.join(savedir, "metrics.npz"),
            **_flatten_metrics_dict(metrics_to_save),
        )

        self.log_resources()

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def split_dataset(self, dset):
        """Identical seeded split to TrainingExperiment.split_dataset, so
        the TUnfold test set matches the Classification-/UnfoldingExperiment
        test set exactly (given the same dataset config)."""

        dcfg = self.cfg.data
        assert dcfg.val_frac > 0, "A validation split is required"
        assert dcfg.test_frac > 0, "A testing split is required"

        fixed_rng = torch.Generator().manual_seed(1729)
        splits = random_split(
            dset,
            [1 - dcfg.val_frac - dcfg.test_frac, dcfg.val_frac, dcfg.test_frac],
            generator=fixed_rng,
        )
        return list(splits)