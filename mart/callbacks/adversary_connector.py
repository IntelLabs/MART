#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import types
from typing import Callable

from lightning.pytorch.callbacks import Callback

from ..utils import MonkeyPatch

__all__ = ["AdversaryConnector"]


class AdversaryConnector(Callback):
    """Perturbs inputs to be adversarial."""

    def __init__(
        self,
        adversary: Callable = None,
        train_adversary: Callable = None,
        val_adversary: Callable = None,
        test_adversary: Callable = None,
        batch_c15n: Callable = None,
    ):
        """A pl.Trainer callback which perturbs input to be adversarial in training/validation/test
        phase.

        Args:
            adversary (Callable, optional): Adversary in the training/validation/test phase if not defined explicitly. Defaults to None.
            train_adversary (Callable, optional): Adversary in the training phase. Defaults to None.
            val_adversary (Callable, optional): Adversary in the validation phase. Defaults to None.
            test_adversary (Callable, optional): Adversary in the test phase. Defaults to None.
            batch_c15n (Callable): Canonicalize batch into convenient format and revert to the original format.
        """
        self.train_adversary = train_adversary or adversary
        self.val_adversary = val_adversary or adversary
        self.test_adversary = test_adversary or adversary
        self.batch_c15n = batch_c15n

    def setup(self, trainer, pl_module, stage=None):
        self._on_after_batch_transfer = pl_module.on_after_batch_transfer
        pl_module.on_after_batch_transfer = types.MethodType(
            self.on_after_batch_transfer, pl_module
        )

    def teardown(self, trainer, pl_module, stage=None):
        pl_module.on_after_batch_transfer = self._on_after_batch_transfer

    def on_after_batch_transfer(self, pl_module, batch, dataloader_idx):
        batch = self._on_after_batch_transfer(batch, dataloader_idx)

        adversary = None

        trainer = pl_module.trainer
        if trainer.training:
            adversary = self.train_adversary
        elif trainer.validating:
            adversary = self.val_adversary
        elif trainer.testing:
            adversary = self.test_adversary

        # Skip if adversary is not defined for all phases train/validation/test.
        if adversary is None:
            return batch

        # Move adversary to same device as pl_module and run attack
        adversary.to(pl_module.device)

        # Make a simple model interface that outputs=model(input, target)
        def model(input, target):
            batch = self.batch_c15n.revert(input, target)

            if hasattr(pl_module, "attack_step"):
                outputs = pl_module.attack_step(batch, dataloader_idx)
            elif hasattr(pl_module, "training_step"):
                # Disable logging if we have to reuse training_step() of the target model.
                with MonkeyPatch(pl_module, "log", lambda *args, **kwargs: None):
                    outputs = pl_module.training_step(batch, dataloader_idx)
            else:
                outputs = model(batch)
            return outputs

        # Canonicalize the batch to work with Adversary.
        input, target = self.batch_c15n(batch)

        adversary.fit(input=input, target=target, model=model)
        input_adv, target_adv = adversary(input=input, target=target)

        # Revert to the original batch format.
        batch_adv = self.batch_c15n.revert(input_adv, target_adv)

        return batch_adv
