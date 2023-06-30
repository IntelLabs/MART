#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import types

from pytorch_lightning.callbacks import Callback

from mart.models import LitModular

__all__ = ["AdversarialTraining"]


class AdversarialTraining(Callback):
    """Perturbs inputs to be adversarial."""

    def __init__(
        self, adversary=None, train_adversary=None, validation_adversary=None, test_adversary=None
    ):
        adversary = adversary or train_adversary

        self.train_adversary = train_adversary or adversary
        self.validation_adversary = validation_adversary or adversary
        self.test_adversary = test_adversary or adversary

    def setup(self, trainer, pl_module, stage=None):
        self._on_after_batch_transfer = pl_module.on_after_batch_transfer
        pl_module.on_after_batch_transfer = types.MethodType(
            self.on_after_batch_transfer, pl_module
        )

    def teardown(self, trainer, pl_module, stage=None):
        pl_module.on_after_batch_transfer = self._on_after_batch_transfer

    def on_after_batch_transfer(self, pl_module, batch, dataloader_idx):
        batch = self._on_after_batch_transfer(batch, dataloader_idx)

        # FIXME: Would be nice if batch was a structured object (or a dict)
        input, target = batch

        trainer = pl_module.trainer
        if trainer.training:
            adversary = self.train_adversary
            step = "training"
        elif trainer.validating:
            adversary = self.validation_adversary
            step = "validation"
        elif trainer.testing:
            adversary = self.test_adversary
            step = "test"
        else:
            return batch

        # Move adversary to same device as pl_module
        adversary.to(pl_module.device)
        input = adversary.attack(pl_module, input=input, target=target, step=step)

        return [input, target]
