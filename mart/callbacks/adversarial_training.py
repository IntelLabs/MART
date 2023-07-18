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
        batch_input_key: str | int = 0,
    ):
        """AdversaryConnector.

        Args:
            adversary (Callable, optional): _description_. Defaults to None.
            train_adversary (Callable, optional): _description_. Defaults to None.
            validation_adversary (Callable, optional): _description_. Defaults to None.
            test_adversary (Callable, optional): _description_. Defaults to None.
            batch_input_key (str | int, optional): Input locator in a batch. Defaults to 0.
        """
        adversary = adversary or train_adversary

        self.train_adversary = train_adversary or adversary
        self.validation_adversary = validation_adversary or adversary
        self.test_adversary = test_adversary or adversary

        self.batch_input_key = batch_input_key

    def setup(self, trainer, pl_module, stage=None):
        self._on_after_batch_transfer = pl_module.on_after_batch_transfer
        pl_module.on_after_batch_transfer = types.MethodType(
            self.on_after_batch_transfer, pl_module
        )

    def teardown(self, trainer, pl_module, stage=None):
        pl_module.on_after_batch_transfer = self._on_after_batch_transfer

    def get_input_target_batcher(self, batch_orig):
        if isinstance(batch_orig, tuple):
            # Convert tuple to list
            batch = list(batch_orig).copy()
        else:
            batch = batch_orig.copy()

        batch_input_key = self.batch_input_key

        # pop() works for both list and dict.
        input = batch.pop(batch_input_key)

        if isinstance(batch, list) and len(batch) == 1:
            target = batch[0]

            def batch_constructor(input, target):
                batch = [target]
                batch.insert(batch_input_key, input)
                return batch

        elif isinstance(batch, list) and len(batch) > 2:
            target = batch.copy()

            def batch_constructor(input, target):
                batch = target.copy()
                batch.insert(batch_input_key, input)
                return batch

        elif isinstance(batch, dict) and "target" in dict:
            target = batch["target"]

            def batch_constructor(input, target):
                return {batch_input_key: input, "target": target}

        elif isinstance(batch, dict) and "target" not in dict:
            # Example in anomalib: dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])
            # image: NCHW;  label: N,
            target = batch

            def batch_constructor(input, target):
                # Besides input and target, add others back to batch.
                return target | {batch_input_key: input}

        else:
            raise NotImplementedError()

        return input, target, batch_constructor

    def wrap_model(self, model, batch_constructor, dataloader_idx):
        """Make a model, such that output = model(input, target)."""

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

        def wrapped_model(*, input, target):
            batch = batch_constructor(input, target)
            output = model_forward(batch)
            return output

        return wrapped_model

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

        # Skip if adversary is not defined for the phase train/validation/test.
        if adversary is None:
            return batch

        # Move adversary to same device as pl_module and run attack
        adversary.to(pl_module.device)

        # FIXME: Directly pass batch instead of assuming it has a structure?
        input, target, batch_constructor = self.get_input_target_batcher(batch)

        # We also need to construct a batch for model during attack iterations.
        model = self.wrap_model(pl_module, batch_constructor, dataloader_idx)

        # TODO: We may need to do model.eval() if there's BN-like layers in the model.
        input_adv = adversary(input=input, target=target, model=model)

        return [input_adv, target]
