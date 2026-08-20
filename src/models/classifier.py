import torch
import torch.nn.functional as F

from src.models.base_model import Model
from src.datasets import UnfoldingData


class Classifier(Model):

    def __init__(self, net, loss="bce"):

        super().__init__(net)

        self.loss = loss
        self.ensembled = self.net.ensembled

    @property
    def lowlevel(self):
        return self.net.lowlevel

    def batch_loss(self, batch: UnfoldingData) -> torch.Tensor:

        # forward pass
        logits = self.forward(batch).squeeze(-1)

        # split by labels to handle imbalanced classes
        mask_sim = batch.labels == 0
        logits_sim = logits[..., mask_sim]  # ellipsis to handle ensembling
        logits_dat = logits[..., ~mask_sim]

        match self.loss:

            case "bce":
                loss_dat = F.binary_cross_entropy_with_logits(
                    logits_dat, torch.ones_like(logits_dat), reduction="none"
                )
                loss_sim = F.binary_cross_entropy_with_logits(
                    logits_sim, torch.zeros_like(logits_sim), reduction="none"
                )
            case "mlc":
                loss_dat = -logits_dat + (-logits_dat).exp()
                loss_sim = logits_sim + logits_sim.exp()
            case _:
                raise ValueError(f"Unknown loss {self.loss}")

        # sample weights
        if batch.sample_logweights is not None:
            sample_weights = batch.sample_logweights.exp()
            loss_dat = loss_dat * sample_weights[~mask_sim]
            loss_sim = loss_sim * sample_weights[mask_sim]

        # ------------------------------------------------------------------
        # Per-replica bootstrap weight (see src.utils.replica.
        # attach_replica_bootstrap): only meaningful/present when training
        # a K-way ENSEMBLED classifier for data statistical uncertainty
        # estimation via replicas. Shape (N, K) -> (K, N) to align with the
        # ensemble dimension already present in `logits` (added via
        # StackedLinear's leading ensemble axis, see MLP.forward). Weight
        # is exactly 1.0 for sim events (labels==0) for every replica --
        # only the data population is bootstrap-varied per replica.
        # ------------------------------------------------------------------
        if batch.replica_weights is not None:
            replica_w = batch.replica_weights.transpose(0, 1)  # (K, N)
            loss_dat = loss_dat * replica_w[..., ~mask_sim]
            loss_sim = loss_sim * replica_w[..., mask_sim]

        # average loss
        mean_dims = (0, 1) if self.ensembled else 0
        loss = (loss_dat.mean(mean_dims) + loss_sim.mean(mean_dims)) / 2

        return loss

    def forward(self, batch: UnfoldingData) -> torch.Tensor:
        """Return the reco-level data-to-sim log-likelihood ratio"""

        if self.lowlevel:
            return self.net(batch.x, c=batch.cond_x, mask=batch.mask_x)

        return self.net(batch.x)

    def weight(self, batch: UnfoldingData) -> torch.Tensor:
        """Return the event-level data-to-sim likelihood ratio"""
        logits = self.forward(batch)
        return logits.exp()

    def prob(self, batch: UnfoldingData) -> torch.Tensor:
        """Return event-level data probabilities"""
        logits = self.forward(batch)
        return logits.sigmoid()


class PartClassifier(Classifier):

    def forward(self, batch: UnfoldingData) -> torch.Tensor:
        """Return the reco-level data-to-sim log-likelihood ratio"""

        if self.lowlevel:
            return self.net(batch.z, c=batch.cond_z, mask=batch.mask_z)

        return self.net(batch.z)