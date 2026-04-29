import logging
import math
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict

log = logging.getLogger("Trainer")


def _format_time(seconds: float) -> str:
    """Format seconds into a human-readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m{secs:02d}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h{mins:02d}m{secs:02d}s"


class _StatusBar:
    """A simple text-based status bar that updates in-place using carriage returns."""

    def __init__(self, description: str = "", total: int = 0, width: int = 40):
        self.description = description
        self.total = total
        self.width = width
        self.current = 0
        self.start_time = time.time()
        self.extra_info = ""

    def update(self, n: int = 1, extra: str = ""):
        self.current += n
        if self.extra_info != extra:
            self.extra_info = extra
            self._redraw()
        else:
            self._redraw()

    def set_total(self, total: int):
        self.total = total
        self._redraw()

    def _redraw(self):
        if self.total == 0:
            return
        elapsed = time.time() - self.start_time
        progress = self.current / self.total
        filled = int(self.width * progress)
        bar = "█" * filled + "░" * (self.width - filled)

        # Estimate time remaining
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current) if self.current < self.total else 0
        else:
            eta = 0

        line = f"\r[{bar}] {progress*100:5.1f}% | {self.current}/{self.total} | ETA: {_format_time(eta)} | elapsed: {_format_time(elapsed)}"
        if self.extra_info:
            line += f" | {self.extra_info}"

        sys.stdout.write(line + " " * 5 + "\r")
        sys.stdout.flush()

    def close(self):
        elapsed = time.time() - self.start_time
        progress = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * progress)
        bar = "█" * filled + "░" * (self.width - filled)
        line = f"\r[{bar}] {progress*100:5.1f}% | {self.current}/{self.total} | elapsed: {_format_time(elapsed)}"
        if self.extra_info:
            line += f" | {self.extra_info}"
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        cfg: DictConfig,
        exp_dir: str,
        device: torch.device,
        use_amp=False,
    ):
        """
        model           -- a pytorch model to be trained
        dataloaders     -- a dictionary containing pytorch data loaders at keys 'train' and 'val'
        cfg             -- configuration dictionary
        exp_dir         -- directory to which training outputs will be saved
        """

        self.model = model
        self.dataloaders = dataloaders
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.device = device
        self.use_amp = use_amp

        self.start_epoch = 0
        self.patience_counter = 0

    def prepare_training(self):

        log.info("Preparing model training")

        # init optimizer
        self.optimizer = instantiate(
            self.cfg.optimizer, params=self.model.trainable_parameters, lr=self.cfg.lr
        )

        # init scaler
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # init scheduler
        self.steps_per_epoch = len(self.dataloaders["train"])
        if self.cfg.scheduler:
            self.scheduler = self.init_scheduler()

        # set logging of metrics
        if self.cfg.use_tensorboard:
            self.summarizer = SummaryWriter(self.exp_dir)
            log.info(f"Writing tensorboard summaries to dir {self.exp_dir}")
        else:
            log.info("`use_tensorboard` set to False. No summaries will be written")

        self.epoch_train_losses = np.array([])
        self.epoch_val_losses = np.array([])
        self.best_val_loss = np.inf

        if self.cfg.warm_start:
            checkpoint = f"model{self.cfg.warm_start_epoch or ''}.pt"
            path = os.path.join(self.exp_dir, checkpoint)
            self.load(path)
            # avoid overriding checkpoint
            os.rename(path, path.replace(".pt", "_old.pt"))
            log.info(f"Warm starting training from epoch {self.start_epoch}")

        # compile model
        self.model = torch.compile(self.model)

    def run_training(self):

        self.prepare_training()

        num_epochs = self.cfg.epochs - self.start_epoch
        log.info(f"Beginning training loop with epochs set to {num_epochs}")
        if self.cfg.patience:
            log.info(f"Early stopping patience set to {self.cfg.patience}")

        t0_total = time.time()

        # Epoch-level status bar
        epoch_bar = _StatusBar(description="Epochs", total=num_epochs)

        # Track epoch timing for estimation
        epoch_times = []

        for e in range(num_epochs):

            self.epoch = (self.start_epoch or 0) + e
            t0_epoch = time.time()

            # train with batch-level status bar
            self.train_one_epoch()

            t_epoch = time.time() - t0_epoch
            epoch_times.append(t_epoch)

            # validate at given frequency
            if (self.epoch + 1) % self.cfg.validate_freq == 0:

                self.validate_one_epoch()

                # check whether validation loss improved
                if (val_loss := self.epoch_val_losses[-1]) < self.best_val_loss:
                    self.patience_counter = 0

                    if self.cfg.save_best_epoch:  # save best checkpoint
                        self.best_val_loss = val_loss
                        self.save()

                elif self.cfg.patience:  # early stopping
                    self.patience_counter += 1
                    if self.patience_counter == self.cfg.patience:
                        epoch_bar.close()
                        log.info(f"Stopping training early at epoch {self.epoch}")
                        break

            # optionally save model at given frequency
            if save_freq := self.cfg.save_freq:
                if (self.epoch + 1) % save_freq == 0 or self.epoch == 0:
                    self.save(tag=self.epoch)

            # Build extra info string with loss and time estimate
            train_loss = self.epoch_train_losses[-1]
            extra_parts = [f"train_loss: {train_loss:.4f}"]

            # Estimate total training time using average of recent epochs
            if len(epoch_times) >= 2:
                avg_epoch_time = np.mean(epoch_times[-min(5, len(epoch_times)):])
                epochs_remaining = num_epochs - self.current_epoch - 1 if hasattr(self, 'current_epoch') else num_epochs - (e + 1)
                eta_total = avg_epoch_time * epochs_remaining
                extra_parts.append(f"ETA: {_format_time(eta_total)}")

            extra_info = " | ".join(extra_parts)
            epoch_bar.update(extra=extra_info)
            self.current_epoch = e

        epoch_bar.close()

        traintime = time.time() - t0_total
        log.info(
            f"Finished training {self.epoch + 1} epochs after {_format_time(traintime)}."
        )

        # save final model
        if not self.cfg.save_best_epoch:
            log.info("Saving final model")
            self.save()

    def train_one_epoch(self):

        # set modules to training mode
        self.model.train()  # NOTE: Ensure frozen submodules are set to eval mode each batch!
        try:
            self.optimizer.train()
        except AttributeError:
            pass

        # create list to save loss per iteration
        train_losses = []

        # Batch-level status bar
        steps_per_epoch = len(self.dataloaders["train"])
        batch_bar = _StatusBar(description=f"Epoch {self.epoch+1}", total=steps_per_epoch, width=30)

        # iterate batch wise over input
        for itr, batch in enumerate(self.dataloaders["train"]):

            batch = batch.to(self.device, non_blocking=True)

            # calculate batch loss
            with torch.autocast(self.device.type, enabled=self.use_amp):
                loss = self.model.batch_loss(batch)

            # update model parameters
            step = itr + self.epoch * self.steps_per_epoch
            total_steps = self.cfg.epochs * self.steps_per_epoch
            self.model.update(
                loss,
                self.optimizer,
                self.scaler,
                step,
                total_steps,
                gradient_norm=self.cfg.gradient_norm,
            )

            # update learning rate
            if self.cfg.scheduler:
                self.scheduler.step()

            # track loss
            train_losses.append(loss.detach())

            # Update batch status bar periodically
            if itr % max(1, steps_per_epoch // 20) == 0 or itr == steps_per_epoch - 1:
                batch_loss_val = float(loss.detach().cpu())
                extra_info = f"loss: {batch_loss_val:.4f}"
                batch_bar.update(extra=extra_info)

            if self.cfg.use_tensorboard and (not step % self.cfg.log_iters) or not step:
                iter_loss = torch.stack(train_losses[-self.cfg.log_iters :])
                self.summarizer.add_scalar(
                    "iter_loss_train",
                    iter_loss.mean().cpu().numpy(),
                    step,
                )
                for k, v in self.model.log_buffer.items():  # model scalars
                    self.summarizer.add_scalar(
                        k,
                        torch.stack(v).mean().cpu().numpy(),
                        step,
                    )
                self.model.log_buffer.clear()

        batch_bar.close()

        # track loss
        self.epoch_train_losses = np.append(
            self.epoch_train_losses, torch.stack(train_losses).mean().cpu().numpy()
        )

        # optionally log to tensorboard
        if self.cfg.use_tensorboard:
            self.summarizer.add_scalar(
                "epoch_loss_train", self.epoch_train_losses[-1], self.epoch
            )
            if self.cfg.scheduler:
                self.summarizer.add_scalar(
                    "learning_rate", self.scheduler.get_last_lr()[0], self.epoch
                )

    @torch.no_grad()
    def validate_one_epoch(self):

        # set modules to evaluation mode
        self.model.eval()
        try:
            self.optimizer.eval()
        except AttributeError:
            pass

        # calculate loss batchwise over input
        val_losses = []
        for batch in self.dataloaders["val"]:

            batch = batch.to(self.device, non_blocking=True)
            # calculate loss
            with torch.autocast(self.device.type, enabled=self.use_amp):
                loss = self.model.batch_loss(batch)
            val_losses.append(loss.detach())

        # track loss
        self.epoch_val_losses = np.append(
            self.epoch_val_losses, torch.stack(val_losses).mean().cpu().numpy()
        )

        # optional logging to tensorboard
        if self.cfg.use_tensorboard:
            self.summarizer.add_scalar(
                "epoch_loss_val", self.epoch_val_losses[-1], self.epoch
            )

    def save(self, tag=""):
        """Save the model along with the training state"""

        # set modules to evaluation mode
        self.model.eval()
        try:
            self.optimizer.eval()
        except AttributeError:
            pass

        model_dict = {
            k.replace("_orig_mod.", ""): v for k, v in self.model.state_dict().items()
        }
        state_dicts = {
            "opt": self.optimizer.state_dict(),
            "model": model_dict,
            "train_losses": self.epoch_train_losses,
            "val_losses": self.epoch_val_losses,
            "epoch": self.epoch,
        }
        if self.cfg.scheduler:
            state_dicts["scheduler"] = self.scheduler.state_dict()
        torch.save(state_dicts, os.path.join(self.exp_dir, f"model{tag}.pt"))

    def load(self, path):
        """Load the model and training state"""

        state_dicts = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dicts["model"])
        if "train_losses" in state_dicts:
            self.epoch_train_losses = state_dicts.get("train_losses", {})
        if "val_losses" in state_dicts:
            self.epoch_val_losses = state_dicts.get("val_losses", {})
            if len(self.epoch_val_losses) > 0:
                self.best_val_loss = self.epoch_val_losses.min()
        if "epoch" in state_dicts:
            self.start_epoch = state_dicts.get("epoch", 0) + 1
        if "opt" in state_dicts:
            self.optimizer.load_state_dict(state_dicts["opt"])
        if "scheduler" in state_dicts:
            self.scheduler.load_state_dict(state_dicts["scheduler"])
        self.model.net.to(self.device)

    def init_scheduler(self):
        scfg = self.cfg.scheduler
        name = scfg._target_
        total_steps = self.cfg.epochs * self.steps_per_epoch
        match name:
            case "torch.optim.lr_scheduler.OneCycleLR":
                return instantiate(
                    scfg,
                    optimizer=self.optimizer,
                    total_steps=total_steps,
                )
            case "torch.optim.lr_scheduler.StepLR":
                return instantiate(
                    scfg,
                    optimizer=self.optimizer,
                    step_size=self.steps_per_epoch * scfg.step_size,
                )
            case _:
                return instantiate(
                    scfg,
                    optimizer=self.optimizer,
                    # total_iters=total_steps,
                )
