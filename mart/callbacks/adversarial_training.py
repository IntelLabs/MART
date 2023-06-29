#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

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

    # FIXME: These are hacks. Ideally we would use on_after_batch_transfer but that isn't exposed to
    #        callbacks only to LightningModules. But maybe we can forward those to callbacks?
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        input, target = batch
        input_adv = self.train_adversary.attack(
            pl_module, input=input, target=target, step="training"
        )
        input[:] = input_adv  # XXX: hacke

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        input, target = batch
        input_adv = self.validation_adversary.attack(
            pl_module, input=input, target=target, step="validation"
        )
        input[:] = input_adv  # XXX: hacke

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        input, target = batch
        input_adv = self.test_adversary.attack(pl_module, input=input, target=target, step="test")
        input[:] = input_adv  # XXX: hacke
