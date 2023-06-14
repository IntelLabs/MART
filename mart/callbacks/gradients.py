#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from collections.abc import Iterable

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import grad_norm

__all__ = ["GradientMonitor"]


class GradientMonitor(Callback):
    def __init__(
        self,
        norm_types: float | int | str | Iterable[float | int | str],
        frequency: int = 100,
        histogram: bool = True,
        clipped: bool = True,
    ):
        if not isinstance(norm_types, Iterable):
            norm_types = [norm_types]

        self.norm_types = norm_types
        self.frequency = frequency
        self.histogram = histogram
        self.clipped = clipped

        self.should_log = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, unused=0):
        self.should_log = batch_idx % self.frequency == 0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        if not self.should_log:
            return

        # Pre-clipping
        self.log_grad_norm(trainer, pl_module, self.norm_types)

        if self.histogram:
            self.log_grad_histogram(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if not self.clipped:
            return

        if not self.should_log:
            return

        # Post-clipping
        postfix = ".clipped_grad"

        self.log_grad_norm(trainer, pl_module, self.norm_types, postfix=postfix)

        if self.histogram:
            self.log_grad_histogram(trainer, pl_module, postfix=postfix)

    def log_grad_norm(self, trainer, pl_module, norm_types, prefix="gradients/", postfix=""):
        for norm_type in self.norm_types:
            norms = grad_norm(pl_module, norm_type)
            norms = {f"{prefix}{key}{postfix}": value for key, value in norms.items()}

            pl_module.log_dict(norms)

    def log_grad_histogram(self, trainer, pl_module, prefix="gradients/", postfix=".grad"):
        for name, param in pl_module.named_parameters():
            if not param.requires_grad:
                continue

            self.log_histogram(trainer, f"{prefix}{name}{postfix}", param.grad)

    def log_histogram(self, trainer, name, values):
        # Add histogram to each logger that supports it
        for logger in trainer.loggers:
            # FIXME: Should we just use isinstance(logger.experiment, SummaryWriter)?
            if not hasattr(logger.experiment, "add_histogram"):
                continue

            logger.experiment.add_histogram(name, values, global_step=trainer.global_step)
