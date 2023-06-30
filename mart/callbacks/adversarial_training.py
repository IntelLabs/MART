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

        # FIXME: Remove use of step
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

        # Create attacked model where the adversary executes before the model
        # FIXME: Should we just use pl_module.training_step? Ideally we would not decompose batch
        #        and instead pass batch directly to the underlying pl_module since it knows how to
        #        interpret batch.
        def attacked_model(input, **batch):
            input_adv = adversary(input=input, **batch)
            return pl_module(input=input_adv, **batch)

        # Move adversary to same device as pl_module and run attack
        # FIXME: Directly pass batch instead of assuming it has a structure?
        input, target = batch
        adversary.to(pl_module.device)
        input_adv = adversary(input=input, target=target, step=step, model=attacked_model)

        return [input_adv, target]
