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

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        input, target = batch

        # FIXME: We reach into LitModular here...how can we get rid of this?
        assert isinstance(pl_module, LitModular)
        model = pl_module.model
        sequence = model._sequences["training"]

        # FIXME: This doesn't work because sequence does not include the Adversary module. How can we fix that?
        #        Because this a callback, we can safely assume the Adversary module should live before the model.
        #        We should be able to "manually" insert it into the sequence here.
        out = self.train_adversary(input=input, target=target, model=model, sequence=sequence)
        print("out =", out)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # FIXME: Copy on_train_batch_start
        pass

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # FIXME: Copy on_train_batch_start
        pass
