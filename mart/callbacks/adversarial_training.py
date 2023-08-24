#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import types
from typing import Callable

from lightning.pytorch.callbacks import Callback

__all__ = ["AdversarialTraining"]


class AdversarialTraining(Callback):
    """Perturbs inputs to be adversarial."""

    def __init__(
        self,
        adversary: Callable = None,
        train_adversary: Callable = None,
        val_adversary: Callable = None,
        test_adversary: Callable = None,
    ):
        """A pl.Trainer callback which perturbs input to be adversarial in training/validation/test
        phase.

        Args:
            adversary (Callable, optional): Adversary in the training/validation/test phase if not defined explicitly. Defaults to None.
            train_adversary (Callable, optional): Adversary in the training phase. Defaults to None.
            val_adversary (Callable, optional): Adversary in the validation phase. Defaults to None.
            test_adversary (Callable, optional): Adversary in the test phase. Defaults to None.
        """
        self.train_adversary = train_adversary or adversary
        self.val_adversary = val_adversary or adversary
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

        # Directly pass batch instead of assuming it has a structure.
        batch_adv = adversary(batch=batch, model=pl_module)

        return batch_adv
