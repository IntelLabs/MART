#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import types

from lightning.pytorch.callbacks import Callback

__all__ = ["AdversarialTraining"]


class AdversarialTraining(Callback):
    """Perturbs inputs to be adversarial."""

    # TODO: training/validation/test or train/val/test
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

        trainer = pl_module.trainer
        if trainer.training:
            adversary = self.train_adversary
        elif trainer.validating:
            adversary = self.validation_adversary
        elif trainer.testing:
            adversary = self.test_adversary
        else:
            return batch

        # Move adversary to same device as pl_module and run attack
        adversary.to(pl_module.device)

        # FIXME: Directly pass batch instead of assuming it has a structure?
        input, target = batch
        input_adv = adversary(input=input, target=target, model=pl_module)

        # Replace the adversarial trainer with the original trainer.
        pl_module.trainer = trainer

        return [input_adv, target]
