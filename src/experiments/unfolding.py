import logging
from contextlib import ExitStack

import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from hydra.utils import instantiate
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from src.experiments.training import TrainingExperiment
from src.utils.metrics import (
    compute_observable_metrics,
    compute_chi2,
    compute_wasserstein,
    compute_mse,
    save_metrics_pdf,
    _flatten_metrics_dict,
)
from src.utils.tunfold import compute_tunfold_result, plot_correlation_matrix
from src.utils.uncertainty import (
    bootstrap_histogram_covariance,
    compute_simple_bin_errors,
    histogram_covariance_from_replicas,
    decompose_data_stat_covariance,
    plot_covariance_matrix,
    plot_relative_error_comparison,
)


# ----------------------------------------------------------------------
# Fixed color palette, so Classifier/AUSSIE/TUnfold always render in the
# same, clearly distinguishable colors regardless of which curves are
# present in a given call (e.g. Classifier skipped on shape mismatch, or
# TUnfold disabled via config) -- avoids relying on plot_reweighting's
# positional default palette, which can silently collide.
# ----------------------------------------------------------------------
COLOR_DATA       = "#323232"
COLOR_SIM        = "#B22222"
COLOR_CLASSIFIER = "#009826"
COLOR_AUSSIE     = "#FFC300"
COLOR_TUNFOLD    = "#1B2A4A"

CURVE_COLOR_MAP = {
    "Classifier": COLOR_CLASSIFIER,
    "AUSSIE": COLOR_AUSSIE,
    "TUnfold": COLOR_TUNFOLD,
}


def _build_colors(names_list):
    """Build an explicit (data_color, sim_color, *curve_colors) list for
    plot_reweighting's `colors` kwarg, keyed off curve name so the
    mapping stays correct regardless of which curves are present."""
    return [COLOR_DATA, COLOR_SIM] + [
        CURVE_COLOR_MAP.get(name, f"C{i}") for i, name in enumerate(names_list)
    ]


class UnfoldingExperiment(TrainingExperiment):

    @torch.inference_mode()
    def evaluate(self, dataloader, tag=None):
        """
        Evaluates the model on the test dataset.
        Predictions are saved alongside truth labels
        """

        self.model.eval()

        predictions = {}

        # get predictions across the test set
        lw_z_sim = []
        for batch in dataloader:

            batch_sim = batch[batch.labels == 0].to(self.device, non_blocking=True)
            lw_sample = (
                0.0
                if batch_sim.sample_logweights is None
                else batch_sim.sample_logweights
            )
            lw_z_sim.append(self.model(batch_sim)[..., 0] + lw_sample)

        if self.model.ensembled:
            predictions["lw_z_sim"] = torch.cat(lw_z_sim, dim=1).cpu()
        else:
            predictions["lw_z_sim"] = torch.cat(lw_z_sim, dim=0).unsqueeze(0).cpu()

        return predictions

    def plot(self):

        if self.cfg.atlas_style:
            from src.utils import plotting_atlasstyle as plotting
        else:
            from src.utils import plotting

        pcfg = self.cfg.plotting
        pw   = pcfg.pagewidth

        savedir = os.path.join(self.exp_dir, "plots")
        os.makedirs(savedir, exist_ok=True)

        # read (untransformed) test data
        self.log.info("Loading test data")
        dset = instantiate(self.cfg.dataset.reader)
        _, _, test_set = self.split_dataset(dset)

        self.log.info("Reading predictions from disk")
        record     = np.load(os.path.join(self.exp_dir, "predictions_test.npz"))
        record_cls = np.load(os.path.join(self.model.cls_path, "predictions_test.npz"))

        labels   = test_set[:].labels
        mask_sim = test_set[:].labels == 0
        mask_dat = ~mask_sim

        lw_z_sim = record["lw_z_sim"]       # [K, N_sim]  already has MC weights from evaluate()
        lw_x     = record_cls["lw_x"]       # [K, N_all]  raw logits, NO MC weights

        classifier_shapes_match = (lw_x.shape[-1] == len(labels))

        if classifier_shapes_match:
            try:
                lw_x_sim = lw_x[..., mask_sim.numpy()].mean(0)   # [N_sim]
            except IndexError:
                self.log.info("Skipping classifier weights due to mismatched shapes")
                lw_x_sim = None
                classifier_shapes_match = False
        else:
            self.log.info(
                f"Classifier predictions size ({lw_x.shape[-1]}) does not "
                f"match current test set size ({len(labels)}) -- skipping "
                f"classifier weights and the classifier-score/AUC plot. "
                f"This usually means the classifier was trained/evaluated "
                f"with a different dataset configuration (num_sim/num_data, "
                f"or a different data population) than the current "
                f"unfolder run."
            )
            lw_x_sim = None

        # ----------------------------------------------------------------
        # Step 1: add iteration weights from previous unfolding step (if any)
        # ----------------------------------------------------------------
        if (p := self.cfg.prev_it_path) is not None:
            lw_sample_sim = torch.from_numpy(
                np.load(os.path.join(p, "unf/predictions_test.npz"))["lw_z_sim"].mean(0)
            )
            if lw_x_sim is not None:
                lw_x_sim += lw_sample_sim.numpy()

        # ----------------------------------------------------------------
        # Step 2: add initial MC sample weights to classifier curve only
        #
        # lw_z_sim -> already includes MC weights from evaluate()  -> do NOT add again
        # lw_x_sim -> raw classifier logits, no MC weights         -> ADD here
        # ----------------------------------------------------------------
        sample_logweights = test_set[:].sample_logweights

        if sample_logweights is not None:

            lw_sample     = sample_logweights.numpy()
            lw_mc_sample  = lw_sample[mask_sim.numpy()]    # [N_sim]
            lw_dat_sample = lw_sample[mask_dat.numpy()]    # [N_dat]

            if lw_x_sim is not None:
                lw_x_sim = lw_x_sim + lw_mc_sample

            sim_weights = np.exp(lw_mc_sample)
            exp_weights = np.exp(lw_dat_sample)

        else:
            exp_weights = None
            sim_weights = None

        # ----------------------------------------------------------------
        # Build weights list for plotting
        #
        # Order: Classifier first, AUSSIE second -- matches desired
        # Classifier=green, AUSSIE=yellow plotting order.
        # ----------------------------------------------------------------
        weights_list  = []
        variance_list = []
        names_list    = []

        if lw_x_sim is not None:
            weights_list.append(np.exp(lw_x_sim))
            variance_list.append(None)
            names_list.append("Classifier")

        weights_list.append(np.exp(lw_z_sim))
        variance_list.append(None)
        names_list.append("AUSSIE")

        # ----------------------------------------------------------------
        # Detect whether the "data" population has particle-level truth
        # info. WWbbData (token-based) exposes mask_z; WWbbMultiData
        # (scalar-feature) does not, so fall back to checking whether z
        # itself is non-trivially non-zero for that population.
        # ----------------------------------------------------------------
        z_dat_check = test_set[:].z[mask_dat]
        mask_z_all = test_set[:].mask_z
        if mask_z_all is not None:
            data_has_truth = bool(mask_z_all[mask_dat].any())
        else:
            data_has_truth = bool((z_dat_check != 0).any())

        if not data_has_truth:
            self.log.info(
                "Data population has no particle-level truth information -- "
                "latents.pdf will show only sim truth curves (unweighted, "
                "classifier-reweighted, AUSSIE-unfolded, TUnfold-unfolded), "
                "without a data-truth reference."
            )

        # ----------------------------------------------------------------
        # Reco-level arrays -- needed both for the "Observables" section
        # below AND for TUnfold's per-observable response matrix inside
        # the latents loop.
        # ----------------------------------------------------------------
        if dset.aux_x is None:
            x_dat = test_set[:].x[mask_dat]
            x_sim = test_set[:].x[mask_sim]
        else:
            x_dat = test_set[:].aux_x[mask_dat]
            x_sim = test_set[:].aux_x[mask_sim]

        # ----------------------------------------------------------------
        # TUnfold config -- resolved ONCE here, with explicit, always-
        # visible logging of whether it's enabled/disabled/missing.
        # ----------------------------------------------------------------
        tcfg = self.cfg.get("tunfold", None)
        tunfold_enabled = tcfg is not None and tcfg.enabled

        if tunfold_enabled:
            self.log.info(
                f"TUnfold reference ENABLED: num_bins_truth={tcfg.num_bins_truth}, "
                f"reco_bin_factor={tcfg.reco_bin_factor}, "
                f"clip_negative={tcfg.clip_negative} -- expect 5 curves "
                f"(Data, Sim, Classifier, AUSSIE, TUnfold) in latents.pdf."
            )
        elif tcfg is None:
            self.log.warning(
                "No `tunfold:` config block found in the resolved config -- "
                "TUnfold curve will be SKIPPED (4 curves only: Data, Sim, "
                "Classifier, AUSSIE). If rerunning a pre-existing run "
                "directory via prev_exp_dir whose saved config predates "
                "this feature, inject it explicitly WITH a '+' prefix "
                "(Hydra's initial composition always validates against "
                "rerun.yaml's bare structure, regardless of what the "
                "target run's own saved config contains), e.g.:\n"
                "  +tunfold.enabled=true +tunfold.num_bins_truth=15 "
                "+tunfold.reco_bin_factor=2 +tunfold.clip_negative=true"
            )
        else:
            self.log.info(
                f"TUnfold explicitly DISABLED (tunfold.enabled={tcfg.enabled}) "
                f"-- 4 curves only: Data, Sim, Classifier, AUSSIE."
            )

        # ----------------------------------------------------------------
        # Latents (gen level)
        # ----------------------------------------------------------------
        self.log.info("Plotting part latents")
        if dset.aux_z is None:
            z_sim = test_set[:].z[mask_sim]
            z_dat = test_set[:].z[mask_dat] if data_has_truth else None
        else:
            z_sim = test_set[:].aux_z[mask_sim]
            z_dat = test_set[:].aux_z[mask_dat] if data_has_truth else None

        # cache per-observable TUnfold weights computed during this loop,
        # so the metrics section below doesn't need to recompute them
        tunfold_weights_by_obs = {}

        with ExitStack() as stack:
            pdf = stack.enter_context(PdfPages(os.path.join(savedir, "latents.pdf")))
            pdf_corr = (
                stack.enter_context(
                    PdfPages(os.path.join(savedir, "tunfold_correlations.pdf"))
                )
                if tunfold_enabled
                else None
            )

            for obs_x, obs in zip(self.process.observables_x, self.process.observables_z):

                # ---- TUnfold: observable-specific weight, appended only
                # to this observable's curve list (unlike Classifier/
                # AUSSIE, whose weight is global and reused across all
                # observables) ----
                weights_list_obs  = list(weights_list)
                variance_list_obs = list(variance_list)
                names_list_obs    = list(names_list)

                if tunfold_enabled:
                    try:
                        result = compute_tunfold_result(
                            x_sim_v=obs_x.compute(x_sim).numpy(),
                            x_dat_v=obs_x.compute(x_dat).numpy(),
                            z_sim_v=obs.compute(z_sim).numpy(),
                            z_dat_v=(obs.compute(z_dat).numpy() if data_has_truth else None),
                            sim_weights=sim_weights,
                            exp_weights=exp_weights,
                            obs_x=obs_x,
                            obs_z=obs,
                            num_bins_truth=tcfg.num_bins_truth,
                            reco_bin_factor=tcfg.reco_bin_factor,
                            clip_negative=tcfg.clip_negative,
                        )
                        tunfold_weights_by_obs[obs.name] = result.weight
                        weights_list_obs.append(result.weight)
                        variance_list_obs.append(result.logvar)
                        names_list_obs.append("TUnfold")

                        fig_corr = plot_correlation_matrix(
                            result.corr, result.truth_edges,
                            xlabel=obs.label,
                            title=f"TUnfold truth-bin correlation: {obs.name}",
                        )
                        pdf_corr.savefig(fig_corr)
                        plt.close(fig_corr)
                    except Exception as e:
                        self.log.warning(
                            f"TUnfold computation failed for observable "
                            f"'{obs.name}' -- skipping TUnfold curve for "
                            f"this observable only. Error: {e}"
                        )

                colors_obs = _build_colors(names_list_obs)

                if data_has_truth:
                    # --- linear bins ---
                    fig, ax = plotting.plot_reweighting(
                        exp=obs.compute(z_dat).numpy(),
                        sim=obs.compute(z_sim).numpy(),
                        weights_list=weights_list_obs,
                        variance_list=variance_list_obs,
                        names_list=names_list_obs,
                        xlabel=obs.label,
                        figsize=np.array([1, 5 / 6]) * pw / 2,
                        num_bins=pcfg.num_bins,
                        discrete=obs.discrete,
                        log_bins=obs.log_bins,
                        logy=obs.logy,
                        qlims=obs.qlims,
                        xlims=obs.xlims,
                        exp_weights=exp_weights,
                        sim_weights=sim_weights,
                        colors=colors_obs,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

                    # --- log bins ---
                    if obs.log_bins:
                        fig, ax = plotting.plot_reweighting(
                            exp=obs.compute(z_dat).numpy(),
                            sim=obs.compute(z_sim).numpy(),
                            weights_list=weights_list_obs,
                            variance_list=variance_list_obs,
                            names_list=names_list_obs,
                            xlabel=obs.label,
                            figsize=np.array([1, 5 / 6]) * pw / 2,
                            num_bins=pcfg.num_bins,
                            discrete=False,
                            log_bins=True,
                            logx=True,
                            logy=obs.logy,
                            qlims=obs.qlims,
                            xlims=None,
                            exp_weights=exp_weights,
                            sim_weights=sim_weights,
                            colors=colors_obs,
                        )
                        pdf.savefig(fig)
                        plt.close(fig)

                else:
                    sim_vals = obs.compute(z_sim).numpy()

                    fig, ax = plotting.plot_reweighting(
                        exp=sim_vals,
                        sim=sim_vals,
                        weights_list=weights_list_obs,
                        variance_list=variance_list_obs,
                        names_list=names_list_obs,
                        xlabel=obs.label,
                        figsize=np.array([1, 5 / 6]) * pw / 2,
                        num_bins=pcfg.num_bins,
                        discrete=obs.discrete,
                        log_bins=obs.log_bins,
                        logy=obs.logy,
                        qlims=obs.qlims,
                        xlims=obs.xlims,
                        exp_weights=None,
                        sim_weights=sim_weights,
                        name_exp="Sim (unweighted)",
                        show_sim=False,
                        add_chi2=False,
                        colors=colors_obs,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

                    if obs.log_bins:
                        fig, ax = plotting.plot_reweighting(
                            exp=sim_vals,
                            sim=sim_vals,
                            weights_list=weights_list_obs,
                            variance_list=variance_list_obs,
                            names_list=names_list_obs,
                            xlabel=obs.label,
                            figsize=np.array([1, 5 / 6]) * pw / 2,
                            num_bins=pcfg.num_bins,
                            discrete=False,
                            log_bins=True,
                            logx=True,
                            logy=obs.logy,
                            qlims=obs.qlims,
                            xlims=None,
                            exp_weights=None,
                            sim_weights=sim_weights,
                            name_exp="Sim (unweighted)",
                            show_sim=False,
                            add_chi2=False,
                            colors=colors_obs,
                        )
                        pdf.savefig(fig)
                        plt.close(fig)

        # ----------------------------------------------------------------
        # Observables (reco level) -- TUnfold has no reco-level curve, so
        # this section is unchanged (Classifier + AUSSIE colors only)
        # ----------------------------------------------------------------
        self.log.info("Plotting reco observables")

        colors_reco = _build_colors(names_list)

        with PdfPages(os.path.join(savedir, "observables.pdf")) as pdf:
            for obs in self.process.observables_x:
                fig, ax = plotting.plot_reweighting(
                    exp=obs.compute(x_dat).numpy(),
                    sim=obs.compute(x_sim).numpy(),
                    weights_list=weights_list,
                    variance_list=variance_list,
                    names_list=names_list,
                    xlabel=obs.label,
                    figsize=np.array([1, 5 / 6]) * pw / 2,
                    num_bins=pcfg.num_bins,
                    discrete=obs.discrete,
                    log_bins=obs.log_bins,
                    logy=obs.logy,
                    qlims=obs.qlims,
                    xlims=obs.xlims,
                    exp_weights=exp_weights,
                    sim_weights=sim_weights,
                    colors=colors_reco,
                )
                pdf.savefig(fig)
                plt.close(fig)

                if obs.log_bins:
                    fig, ax = plotting.plot_reweighting(
                        exp=obs.compute(x_dat).numpy(),
                        sim=obs.compute(x_sim).numpy(),
                        weights_list=weights_list,
                        variance_list=variance_list,
                        names_list=names_list,
                        xlabel=obs.label,
                        figsize=np.array([1, 5 / 6]) * pw / 2,
                        num_bins=pcfg.num_bins,
                        discrete=False,
                        log_bins=True,
                        logx=True,
                        logy=obs.logy,
                        qlims=obs.qlims,
                        xlims=None,
                        exp_weights=exp_weights,
                        sim_weights=sim_weights,
                        colors=colors_reco,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

            # ---- classifier score plot -------------------------------------
            if classifier_shapes_match:
                labels_np      = labels.int().numpy()
                wz             = np.exp(lw_z_sim).mean(0)
                lw_x_mean      = lw_x.mean(0)
                preds          = expit(lw_x_mean)
                sample_weights = np.ones(len(labels_np))
                sample_weights[labels_np == 0] = wz
                if exp_weights is not None:
                    sample_weights[labels_np == 1] = exp_weights

                fig, ax = plotting.plot_reweighting(
                    exp=lw_x_mean[mask_dat],
                    sim=lw_x_mean[mask_sim],
                    weights_list=[wz],
                    variance_list=[variance_list[-1]],
                    names_list=[names_list[-1]],
                    figsize=np.array([1, 5 / 6]) * pw / 2,
                    num_bins=pcfg.num_bins,
                    discrete=obs.discrete,
                    exp_weights=exp_weights,
                    sim_weights=sim_weights,
                    xlabel=r"$\log R_\theta(x)$",
                    logy=True,
                    qlims=(1e-5, 1 - 1e-5),
                    density=True,
                    ratio_lims=(0.6, 1.4),
                    colors=_build_colors([names_list[-1]]),
                )

                plt.subplots_adjust(top=0.9)
                auc = roc_auc_score(labels_np, preds, sample_weight=sample_weights)
                fig.suptitle(f"AUC = {auc:.5f}")

                pdf.savefig(fig)
                plt.close(fig)
            else:
                self.log.info(
                    "Skipping classifier-score/AUC plot due to mismatched "
                    "shapes between classifier predictions and current "
                    "test set."
                )

        # ----------------------------------------------------------------
        # Evaluation metrics summary (chi2, Wasserstein, MSE per observable)
        # ----------------------------------------------------------------
        self.log.info("Computing evaluation metrics")

        obs_names_x, metrics_x = compute_observable_metrics(
            self.process.observables_x,
            x_dat, x_sim,
            exp_weights, sim_weights,
            weights_list, names_list,
            num_bins=pcfg.num_bins,
            name_sim="MC Simulation Pythia",
        )

        metrics_to_save = {"observables_x": {"names": obs_names_x, **metrics_x}}

        with PdfPages(os.path.join(savedir, "metrics.pdf")) as pdf:
            save_metrics_pdf(pdf, obs_names_x, metrics_x, prefix="Reco-level: ")

            if data_has_truth:
                obs_names_z, metrics_z = compute_observable_metrics(
                    self.process.observables_z,
                    z_dat, z_sim,
                    exp_weights, sim_weights,
                    weights_list, names_list,
                    num_bins=pcfg.num_bins,
                    name_sim="MC Simulation Pythia",
                )

                # ---- add TUnfold to the truth-level metrics summary,
                # reusing weights already computed in the latents loop ----
                if tunfold_enabled:
                    metrics_z["chi2"]["TUnfold"] = []
                    metrics_z["wasserstein"]["TUnfold"] = []
                    metrics_z["mse"]["TUnfold"] = []
                    for obs in self.process.observables_z:
                        z_dat_v = obs.compute(z_dat).numpy()
                        z_sim_v = obs.compute(z_sim).numpy()
                        tunfold_weight = tunfold_weights_by_obs.get(obs.name)

                        if tunfold_weight is None:
                            metrics_z["chi2"]["TUnfold"].append(np.nan)
                            metrics_z["wasserstein"]["TUnfold"].append(np.nan)
                            metrics_z["mse"]["TUnfold"].append(np.nan)
                            continue

                        metrics_z["chi2"]["TUnfold"].append(
                            compute_chi2(
                                z_dat_v, z_sim_v, exp_weights, tunfold_weight,
                                num_bins=pcfg.num_bins, qlims=obs.qlims, xlims=obs.xlims,
                            )
                        )
                        metrics_z["wasserstein"]["TUnfold"].append(
                            compute_wasserstein(z_dat_v, z_sim_v, exp_weights, tunfold_weight)
                        )
                        metrics_z["mse"]["TUnfold"].append(
                            compute_mse(
                                z_dat_v, z_sim_v, exp_weights, tunfold_weight,
                                num_bins=pcfg.num_bins, qlims=obs.qlims, xlims=obs.xlims,
                            )
                        )

                save_metrics_pdf(pdf, obs_names_z, metrics_z, prefix="Truth-level: ")
                metrics_to_save["observables_z"] = {"names": obs_names_z, **metrics_z}

        np.savez(
            os.path.join(savedir, "metrics.npz"),
            **_flatten_metrics_dict(metrics_to_save),
        )

        # ------------------------------------------------------------------
        # Statistical uncertainty
        #
        #   1. MC-stat / data-reference-stat (bootstrap_histogram_covariance):
        #      cheap, retraining-free, using a configurable bootstrap
        #      distribution (lognormal by default, poisson optional).
        #   2. Data-bootstrap-varying ensemble spread
        #      (histogram_covariance_from_replicas on THIS run's K
        #      replicas): conflates genuine data-statistical uncertainty
        #      with training/initialization stochasticity -- the
        #      OmniFold-paper-style bootstrap procedure.
        #   3. If `uncertainty.control_exp_dir` points at a matching
        #      CONTROL run (same architecture/K, replica_bootstrap.
        #      enabled=false), subtract its ensemble spread from (2) to
        #      isolate the genuine DATA-statistical component.
        #   4. Relative-error comparison plots (simple sqrt(sum w^2)
        #      formula vs. sqrt(diag(bootstrap covariance))) -- a
        #      closure/validation check for (1), doubling as a direct
        #      visual comparison of relative precision per bin.
        # ------------------------------------------------------------------
        ucfg = self.cfg.get("uncertainty", None)
        uncertainty_enabled = ucfg is not None and ucfg.enabled

        rcfg = self.cfg.get("replica_bootstrap", None)
        replicas_available = (
            rcfg is not None and rcfg.enabled and lw_z_sim.shape[0] > 1
        )

        control_exp_dir = ucfg.get("control_exp_dir", None) if uncertainty_enabled else None
        control_available = False
        lw_z_sim_control = None
        lw_x_control = None

        if replicas_available and control_exp_dir:
            control_pred_path = os.path.join(control_exp_dir, "predictions_test.npz")
            control_cls_path = None
            if os.path.exists(control_pred_path):
                record_control = np.load(control_pred_path)
                lw_z_sim_control = record_control["lw_z_sim"]

                control_cfg_path = os.path.join(control_exp_dir, ".hydra", "config.yaml")
                if os.path.exists(control_cfg_path):
                    from omegaconf import OmegaConf
                    control_cfg = OmegaConf.load(control_cfg_path)
                    control_cls_path = control_cfg.model.get("cls_path", None)

                if control_cls_path is not None:
                    control_cls_pred_path = os.path.join(control_cls_path, "predictions_test.npz")
                    if os.path.exists(control_cls_pred_path):
                        lw_x_control = np.load(control_cls_pred_path)["lw_x"]

                if lw_z_sim_control.shape[0] == lw_z_sim.shape[0]:
                    control_available = True
                    self.log.info(
                        f"Control ensemble FOUND at {control_exp_dir} "
                        f"(K={lw_z_sim_control.shape[0]}) -- will subtract "
                        f"training-stochasticity variance to isolate genuine "
                        f"DATA-statistical uncertainty."
                    )
                else:
                    self.log.warning(
                        f"Control ensemble at {control_exp_dir} has "
                        f"K={lw_z_sim_control.shape[0]}, but this run has "
                        f"K={lw_z_sim.shape[0]} -- skipping subtraction "
                        f"(K must match)."
                    )
            else:
                self.log.warning(
                    f"uncertainty.control_exp_dir={control_exp_dir} does not "
                    f"contain predictions_test.npz -- skipping subtraction."
                )

        if uncertainty_enabled:

            boot_distribution = ucfg.get("distribution", "lognormal")
            boot_lognormal_sigma = ucfg.get("lognormal_sigma", 1.0)

            self.log.info(
                f"Statistical uncertainty ENABLED: n_boot={ucfg.n_boot}, "
                f"distribution={boot_distribution}"
                f"{f' (sigma={boot_lognormal_sigma})' if boot_distribution == 'lognormal' else ''} "
                f"(MC-stat / data-reference-stat, retraining-free). "
                f"Data-bootstrap-varying ensemble (conflated with training "
                f"stochasticity): {'AVAILABLE (K=' + str(lw_z_sim.shape[0]) + ')' if replicas_available else 'NOT AVAILABLE'}. "
                f"Clean data-stat via control subtraction: "
                f"{'AVAILABLE' if control_available else 'NOT AVAILABLE'}. "
                f"Writing plots/uncertainty_matrices.pdf"
            )

            with PdfPages(os.path.join(savedir, "uncertainty_matrices.pdf")) as pdf_unc:
                for obs in self.process.observables_z:

                    z_sim_v = obs.compute(z_sim).numpy()
                    lo, hi = obs.xlims or tuple(
                        np.quantile(z_sim_v, obs.qlims or (0.005, 0.995))
                    )
                    if obs.log_bins:
                        lo = max(lo, 1e-10)
                        bins = np.logspace(np.log10(lo), np.log10(hi), pcfg.num_bins)
                    else:
                        bins = np.linspace(lo, hi, pcfg.num_bins)

                                        # ---- "before unfolding" reference: raw sim weight,
                    # same binning, for the relative-error comparison plot ----
                    sim_w_for_compare = sim_weights if sim_weights is not None else np.ones_like(z_sim_v)
                    mean_before, err_before = compute_simple_bin_errors(z_sim_v, sim_w_for_compare, bins)

                    # ---- 1. MC-stat (retraining-free), one per reweighted curve ----
                    for weight, name in zip(weights_list, names_list):
                        w = weight.mean(0) if np.ndim(weight) > 1 else weight
                        mean, cov, corr = bootstrap_histogram_covariance(
                            z_sim_v, w, bins, n_boot=ucfg.n_boot, seed=ucfg.seed,
                            distribution=boot_distribution,
                            lognormal_sigma=boot_lognormal_sigma,
                        )

                        fig_cov = plot_covariance_matrix(
                            cov, bins, xlabel=obs.label,
                            title=f"{name} MC-stat covariance: {obs.name} (n_boot={ucfg.n_boot})",
                        )
                        pdf_unc.savefig(fig_cov)
                        plt.close(fig_cov)

                        fig_corr = plot_correlation_matrix(
                            corr, bins, xlabel=obs.label,
                            title=f"{name} MC-stat correlation: {obs.name} (n_boot={ucfg.n_boot})",
                        )
                        pdf_unc.savefig(fig_corr)
                        plt.close(fig_corr)

                        # ---- relative-error closure/comparison plot,
                        # now including the pre-unfolding reference ----
                        mean_simple, err_simple = compute_simple_bin_errors(z_sim_v, w, bins)
                        err_cov = np.sqrt(np.clip(np.diag(cov), 0, None))
                        fig_rel = plot_relative_error_comparison(
                            bins, mean_simple, err_simple, err_cov, xlabel=obs.label,
                            mean_before=mean_before, err_before=err_before,
                            title=f"{name} relative statistical error: {obs.name}",
                        )
                        pdf_unc.savefig(fig_rel)
                        plt.close(fig_rel)

                    # ---- data / pseudodata REFERENCE histogram stat unc. ----
                    if data_has_truth:
                        z_dat_v = obs.compute(z_dat).numpy()
                        ew = exp_weights if exp_weights is not None else np.ones_like(z_dat_v)
                        mean_d, cov_d, corr_d = bootstrap_histogram_covariance(
                            z_dat_v, ew, bins, n_boot=ucfg.n_boot, seed=ucfg.seed,
                            distribution=boot_distribution,
                            lognormal_sigma=boot_lognormal_sigma,
                        )

                        fig_cov = plot_covariance_matrix(
                            cov_d, bins, xlabel=obs.label,
                            title=f"Data-reference stat covariance: {obs.name} (n_boot={ucfg.n_boot})",
                        )
                        pdf_unc.savefig(fig_cov)
                        plt.close(fig_cov)

                        fig_corr = plot_correlation_matrix(
                            corr_d, bins, xlabel=obs.label,
                            title=f"Data-reference stat correlation: {obs.name} (n_boot={ucfg.n_boot})",
                        )
                        pdf_unc.savefig(fig_corr)
                        plt.close(fig_corr)

                        mean_simple_d, err_simple_d = compute_simple_bin_errors(z_dat_v, ew, bins)
                        err_cov_d = np.sqrt(np.clip(np.diag(cov_d), 0, None))
                        fig_rel = plot_relative_error_comparison(
                            bins, mean_simple_d, err_simple_d, err_cov_d, xlabel=obs.label,
                            title=f"Data-reference relative statistical error: {obs.name}",
                        )
                        pdf_unc.savefig(fig_rel)
                        plt.close(fig_rel)

                    # ---- 2. data-bootstrap-varying ensemble (CONFLATED, OmniFold-paper-style) ----
                    if replicas_available:
                        K = lw_z_sim.shape[0]
                        aussie_replica_weights = np.exp(lw_z_sim)  # (K, N_sim)

                        mean_r, cov_r, corr_r = histogram_covariance_from_replicas(
                            z_sim_v, aussie_replica_weights, bins,
                        )

                        fig_cov = plot_covariance_matrix(
                            cov_r, bins, xlabel=obs.label,
                            title=f"AUSSIE bootstrap-ensemble covariance (CONFLATED): {obs.name} (K={K})",
                        )
                        pdf_unc.savefig(fig_cov)
                        plt.close(fig_cov)

                        fig_corr = plot_correlation_matrix(
                            corr_r, bins, xlabel=obs.label,
                            title=f"AUSSIE bootstrap-ensemble correlation (CONFLATED): {obs.name} (K={K})",
                        )
                        pdf_unc.savefig(fig_corr)
                        plt.close(fig_corr)

                        # ---- 3. control-subtracted, CLEAN data-stat ----
                        if control_available:
                            control_replica_weights = np.exp(lw_z_sim_control)
                            mean_c, cov_c, corr_c = histogram_covariance_from_replicas(
                                z_sim_v, control_replica_weights, bins,
                            )

                            cov_data, corr_data, n_neg = decompose_data_stat_covariance(
                                cov_r, cov_c, log=self.log,
                                label=f"AUSSIE / {obs.name}",
                            )

                            fig_cov = plot_covariance_matrix(
                                cov_c, bins, xlabel=obs.label,
                                title=f"AUSSIE control-ensemble covariance (training-only): {obs.name} (K={K})",
                            )
                            pdf_unc.savefig(fig_cov)
                            plt.close(fig_cov)

                            fig_cov = plot_covariance_matrix(
                                cov_data, bins, xlabel=obs.label,
                                title=f"AUSSIE CLEAN data-stat covariance (subtracted): {obs.name} "
                                      f"(K={K}, {n_neg} bins clipped)",
                            )
                            pdf_unc.savefig(fig_cov)
                            plt.close(fig_cov)

                            fig_corr = plot_correlation_matrix(
                                corr_data, bins, xlabel=obs.label,
                                title=f"AUSSIE CLEAN data-stat correlation (subtracted): {obs.name} (K={K})",
                            )
                            pdf_unc.savefig(fig_corr)
                            plt.close(fig_corr)

                        if classifier_shapes_match and lw_x.shape[0] > 1:
                            cls_replica_weights = np.exp(lw_x[..., mask_sim.numpy()])
                            mean_rc, cov_rc, corr_rc = histogram_covariance_from_replicas(
                                z_sim_v, cls_replica_weights, bins,
                            )

                            fig_cov = plot_covariance_matrix(
                                cov_rc, bins, xlabel=obs.label,
                                title=f"Classifier bootstrap-ensemble covariance (CONFLATED): {obs.name} (K={K})",
                            )
                            pdf_unc.savefig(fig_cov)
                            plt.close(fig_cov)

                            fig_corr = plot_correlation_matrix(
                                corr_rc, bins, xlabel=obs.label,
                                title=f"Classifier bootstrap-ensemble correlation (CONFLATED): {obs.name} (K={K})",
                            )
                            pdf_unc.savefig(fig_corr)
                            plt.close(fig_corr)

                            if control_available and lw_x_control is not None and lw_x_control.shape[0] > 1:
                                cls_control_weights = np.exp(lw_x_control[..., mask_sim.numpy()])
                                mean_ccc, cov_ccc, corr_ccc = histogram_covariance_from_replicas(
                                    z_sim_v, cls_control_weights, bins,
                                )
                                cov_cls_data, corr_cls_data, n_neg_cls = decompose_data_stat_covariance(
                                    cov_rc, cov_ccc, log=self.log,
                                    label=f"Classifier / {obs.name}",
                                )

                                fig_cov = plot_covariance_matrix(
                                    cov_cls_data, bins, xlabel=obs.label,
                                    title=f"Classifier CLEAN data-stat covariance (subtracted): {obs.name} "
                                          f"(K={K}, {n_neg_cls} bins clipped)",
                                )
                                pdf_unc.savefig(fig_cov)
                                plt.close(fig_cov)

                                fig_corr = plot_correlation_matrix(
                                    corr_cls_data, bins, xlabel=obs.label,
                                    title=f"Classifier CLEAN data-stat correlation (subtracted): {obs.name} (K={K})",
                                )
                                pdf_unc.savefig(fig_corr)
                                plt.close(fig_corr)

        elif ucfg is None:
            self.log.info(
                "No `uncertainty:` config block found -- skipping statistical "
                "uncertainty matrices."
            )
        else:
            self.log.info("Statistical uncertainty matrices explicitly disabled.")