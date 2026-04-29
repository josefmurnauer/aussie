import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from hydra.utils import instantiate
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from src.experiments.training import TrainingExperiment
from src.utils import plotting


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

        try:
            lw_x_sim = lw_x[..., mask_sim.numpy()].mean(0)   # [N_sim]
        except IndexError:
            self.log.info("Skipping classifier weights due to mismatched shapes")
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
        # ----------------------------------------------------------------
        weights_list  = [np.exp(lw_z_sim)]
        variance_list = [None]
        names_list    = ["AUSSIE"]

        if lw_x_sim is not None:
            weights_list.append(np.exp(lw_x_sim))
            variance_list.append(None)
            names_list.append("Classifier")

        # ----------------------------------------------------------------
        # Latents (gen level)
        # ----------------------------------------------------------------
        self.log.info("Plotting part latents")
        if dset.aux_z is None:
            z_dat = test_set[:].z[mask_dat]
            z_sim = test_set[:].z[mask_sim]
        else:
            z_dat = test_set[:].aux_z[mask_dat]
            z_sim = test_set[:].aux_z[mask_sim]

        with PdfPages(os.path.join(savedir, "latents.pdf")) as pdf:
            for obs in self.process.observables_z:
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
                # --- log bins ---
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
                    xlims=obs.xlims,
                    exp_weights=exp_weights,
                    sim_weights=sim_weights,    # <- Sim histogram weighted
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
                # --- log bins ---
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
                    xlims=obs.xlims,
                    exp_weights=exp_weights,
                    sim_weights=sim_weights,    # <- Sim histogram weighted
                )
                pdf.savefig(fig)
                plt.close(fig)

            # ---- classifier score plot ------------------------------------
            labels         = labels.int().numpy()
            wz             = np.exp(lw_z_sim).mean(0)
            lw_x_mean      = lw_x.mean(0)
            preds          = expit(lw_x_mean)
            sample_weights = np.ones(len(labels))
            sample_weights[labels == 0] = wz
            if exp_weights is not None:
                sample_weights[labels == 1] = exp_weights

            fig, ax = plotting.plot_reweighting(
                exp=lw_x_mean[mask_dat],
                sim=lw_x_mean[mask_sim],
                weights_list=[wz],
                variance_list=[variance_list[0]],
                names_list=[names_list[0]],
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
            auc = roc_auc_score(labels, preds, sample_weight=sample_weights)
            fig.suptitle(f"AUC = {auc:.5f}")

            pdf.savefig(fig)
            plt.close(fig)