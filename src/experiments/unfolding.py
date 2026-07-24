import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from hydra.utils import instantiate
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from src.experiments.training import TrainingExperiment
from src.utils.metrics import compute_observable_metrics, save_metrics_pdf


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
            # lw_z_sim.append(self.model(batch_sim).squeeze(-1) + lw_sample)
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

        lw_z_sim = record["lw_z_sim"]       # [1, N_sim]  already has MC weights from evaluate()
        lw_x     = record_cls["lw_x"]       # [1, N_all]  raw logits, NO MC weights

        # ----------------------------------------------------------------
        # The classifier's saved predictions (record_cls) may not match
        # the size of the CURRENT unfolder run's test set -- e.g. if the
        # classifier was trained/evaluated with a different num_sim/
        # num_data, or before a real-data population was added/removed.
        # Detect this once and gate BOTH the reweighting-curve usage AND
        # the classifier-score/AUC plot on it, to avoid a crash later.
        # ----------------------------------------------------------------
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
        #
        # log space addition = linear space multiplication:
        #     final_weight = classifier_weight * mc_sample_weight
        # ----------------------------------------------------------------
        sample_logweights = test_set[:].sample_logweights

        if sample_logweights is not None:

            lw_sample     = sample_logweights.numpy()
            lw_mc_sample  = lw_sample[mask_sim.numpy()]    # [N_sim]
            lw_dat_sample = lw_sample[mask_dat.numpy()]    # [N_dat]

            # add MC sample weights to classifier curve (not to AUSSIE/lw_z_sim)
            if lw_x_sim is not None:
                lw_x_sim = lw_x_sim + lw_mc_sample

            # sim_weights for the unweighted Sim histogram
            sim_weights = np.exp(lw_mc_sample)

            # data sample weights for plotting data histogram
            exp_weights = np.exp(lw_dat_sample)

        else:
            exp_weights = None
            sim_weights = None

        # ----------------------------------------------------------------
        # Build weights list for plotting
        # lw_z_sim already has MC weights -> use directly
        # lw_x_sim now has MC weights added above
        #
        # Order: Classifier first (gets the 3rd color slot in the palette),
        # AUSSIE second (gets the 4th color slot) -- matches desired
        # Classifier=green, AUSSIE=next-color plotting order.
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
        # itself is non-trivially non-zero for that population. This makes
        # the plotting logic work correctly for BOTH:
        #   - Herwig pseudodata (MC, has truth)
        #   - real collision data (no truth at all)
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
                "classifier-reweighted, AUSSIE-unfolded), without a "
                "data-truth reference."
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

        with PdfPages(os.path.join(savedir, "latents.pdf")) as pdf:
            for obs in self.process.observables_z:

                if data_has_truth:
                    # --- linear bins ---
                    fig, ax = plotting.plot_reweighting(
                        exp=obs.compute(z_dat).numpy(),
                        sim=obs.compute(z_sim).numpy(),
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
                        sim_weights=sim_weights,    # <- Sim histogram weighted
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

                    # --- log bins (only for observables explicitly marked
                    # as suitable for it) ---
                    if obs.log_bins:
                        fig, ax = plotting.plot_reweighting(
                            exp=obs.compute(z_dat).numpy(),
                            sim=obs.compute(z_sim).numpy(),
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
                            sim_weights=sim_weights,    # <- Sim histogram weighted
                        )
                        pdf.savefig(fig)
                        plt.close(fig)

                else:
                    # --- no data truth available: plot sim-only, showing
                    # unweighted, classifier-reweighted, and AUSSIE-unfolded
                    # truth curves. exp and sim are both set to the sim
                    # array, with show_sim=False so the raw "Sim" curve
                    # isn't duplicated, and add_chi2=False since
                    # sim-vs-itself is meaningless.
                    sim_vals = obs.compute(z_sim).numpy()

                    fig, ax = plotting.plot_reweighting(
                        exp=sim_vals,
                        sim=sim_vals,
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
                        exp_weights=None,
                        sim_weights=sim_weights,
                        name_exp="Sim (unweighted)",
                        show_sim=False,
                        add_chi2=False,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

                    if obs.log_bins:
                        fig, ax = plotting.plot_reweighting(
                            exp=sim_vals,
                            sim=sim_vals,
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
                            exp_weights=None,
                            sim_weights=sim_weights,
                            name_exp="Sim (unweighted)",
                            show_sim=False,
                            add_chi2=False,
                        )
                        pdf.savefig(fig)
                        plt.close(fig)

        # ----------------------------------------------------------------
        # Observables (reco level)
        # ----------------------------------------------------------------
        self.log.info("Plotting reco observables")
        if dset.aux_x is None:
            x_dat = test_set[:].x[mask_dat]
            x_sim = test_set[:].x[mask_sim]
        else:
            x_dat = test_set[:].aux_x[mask_dat]
            x_sim = test_set[:].aux_x[mask_sim]

        with PdfPages(os.path.join(savedir, "observables.pdf")) as pdf:
            for obs in self.process.observables_x:
                # --- linear bins ---
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
                    sim_weights=sim_weights,    # <- Sim histogram weighted
                )
                pdf.savefig(fig)
                plt.close(fig)

                # --- log bins (see guard explanation above) ---
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
                        sim_weights=sim_weights,    # <- Sim histogram weighted
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

            # ---- classifier score plot -------------------------------------
            # Only meaningful/possible if the classifier's saved predictions
            # actually match the current test set (see classifier_shapes_match
            # computed above). Otherwise skip entirely rather than crash.
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
                    sim_weights=sim_weights,        # <- Sim histogram weighted
                    xlabel=r"$\log R_\theta(x)$",
                    logy=True,
                    qlims=(1e-5, 1 - 1e-5),
                    density=True,
                    ratio_lims=(0.6, 1.4),
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
            weights_list, names_list,   # already built above: Classifier (+ AUSSIE)
            num_bins=pcfg.num_bins,
            name_sim="MC Simulation Pythia",
        )

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
                save_metrics_pdf(pdf, obs_names_z, metrics_z, prefix="Truth-level: ")