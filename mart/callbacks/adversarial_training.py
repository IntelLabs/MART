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

__all__ = ["AdversarialTraining"]


class AdversarialTraining(Callback):
    """Perturbs inputs to be adversarial."""

    # TODO: training/validation/test or train/val/test
    def __init__(
        self,
        adversary: Callable = None,
        train_adversary: Callable = None,
        validation_adversary: Callable = None,
        test_adversary: Callable = None,
    ):
        """AdversaryConnector.

        Args:
            adversary (Callable, optional): _description_. Defaults to None.
            train_adversary (Callable, optional): _description_. Defaults to None.
            validation_adversary (Callable, optional): _description_. Defaults to None.
            test_adversary (Callable, optional): _description_. Defaults to None.
        """
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

    def wrap_model(self, model, dataloader_idx):
        """Make a model, such that `output = model(batch)`."""

        # Consume dataloader_idx
        if hasattr(model, "attack_step"):

            def model_forward(batch):
                output = model.attack_step(batch, dataloader_idx)
                return output

        elif hasattr(model, "training_step"):
            # Monkey-patch model.log to avoid spamming.
            @MonkeyPatch(model, "log", lambda *args, **kwargs: None)
            def model_forward(batch):
                output = model.training_step(batch, dataloader_idx)
                return output

        else:
            model_forward = model

        return model_forward

    def on_after_batch_transfer(self, pl_module, batch, dataloader_idx):
        batch = self._on_after_batch_transfer(batch, dataloader_idx)

        adversary = None

        trainer = pl_module.trainer
        if trainer.training:
            adversary = self.train_adversary
        elif trainer.validating:
            adversary = self.validation_adversary
        elif trainer.testing:
            adversary = self.test_adversary

        # Skip if adversary is not defined for all phases train/validation/test.
        if adversary is None:
            return batch

        # Move adversary to same device as pl_module and run attack
        adversary.to(pl_module.device)

        # We assume Adversary is not aware of PyTorch Lightning,
        # so wrap the model as `output=model(batch)`.
        model = self.wrap_model(pl_module, dataloader_idx)

        # TODO: We may need to do model.eval() if there's BN-like layers in the model.
        # Directly pass batch instead of assuming it has a structure.
        batch_adv = adversary(batch=batch, model=model)

        return batch_adv
