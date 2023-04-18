#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import os

from pytorch_lightning.callbacks import Callback
from torchvision.transforms import ToPILImage

__all__ = ["PerturbedImageVisualizer", "PerturbationVisualizer"]


class PerturbedImageVisualizer(Callback):
    """Save adversarial images as files."""

    def __init__(self, folder):
        super().__init__()

        # FIXME: This should use the Trainer's logging directory.
        self.folder = folder
        self.convert = ToPILImage()

        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        # Save input and target for on_train_end
        self.input = batch["input"]
        self.target = batch["target"]

    def on_train_end(self, trainer, model):
        # FIXME: We should really just save this to outputs instead of recomputing adv_input
        adv_input = model(input=self.input, target=self.target)

        for img, tgt in zip(adv_input, self.target):
            fname = tgt["file_name"]
            fpath = os.path.join(self.folder, fname)
            im = self.convert(img / 255)
            im.save(fpath)


class PerturbationVisualizer(Callback):
    def __init__(self, frequency: int = 100):
        self.frequency = frequency

    def log_perturbation(self, trainer, pl_module):
        # FIXME: Generalize this by using DotDict?
        perturbation = pl_module.model.perturber.perturbation

        # Add image to each logger
        for logger in trainer.loggers:
            # FIXME: Should we just use isinstance(logger.experiment, SummaryWriter)?
            if not hasattr(logger.experiment, "add_image"):
                continue

            logger.experiment.add_image(
                "perturbation", perturbation, global_step=trainer.global_step
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.frequency != 0:
            return

        self.log_perturbation(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self.log_perturbation(trainer, pl_module)
