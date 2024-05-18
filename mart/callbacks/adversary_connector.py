#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import types
from typing import Any, Callable

import torch
from lightning.pytorch.callbacks import Callback

from ..utils import MonkeyPatch, get_pylogger

logger = get_pylogger(__name__)


__all__ = ["AdversaryConnector"]


class training_mode:
    """A context that switches a torch.nn.Module object to the training mode except for some
    children."""

    def __init__(self, module, excludes=[]):
        self.module = module
        self.excludes = excludes

    def __enter__(self):
        # Save the original training mode status.
        self.training = self.module.training
        self.module.train(True)
        # Set some children modules of "excludes" to eval mode instead.
        self.selective_eval_mode("", self.module, self.excludes)

    def selective_eval_mode(self, path, model, eval_mode_module_names):
        if model.__module__ in eval_mode_module_names:
            model.eval()
            logger.debug(f"Set {path}: {model.__class__.__name__} to eval mode.")
        else:
            for child_name, child in model.named_children():
                if isinstance(model, torch.nn.Sequential):
                    child_path = f"{path}[{child_name}]"
                else:
                    child_path = f"{path}.{child_name}"
                self.selective_eval_mode(child_path, child, eval_mode_module_names)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        # Restore the original training mode status.
        self.module.train(self.training)


class AdversaryConnector(Callback):
    """Perturbs inputs to be adversarial."""

    def __init__(
        self,
        adversary: Callable = None,
        train_adversary: Callable = None,
        val_adversary: Callable = None,
        test_adversary: Callable = None,
        batch_c15n: Callable = None,
        module_step_fn: str = "training_step",
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
        self.module_step_fn = module_step_fn

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
                # Users can implement LightningModule:attack_step() for generating adversarial examples.
                outputs = pl_module.attack_step(batch, dataloader_idx)
            else:
                # If there is no attack_step(), we will borrow LightningModule:training_step() with some modifications.
                #   1. Disable the training logging mechanism.
                #   2. Switch to the training mode to get the loss value, except for children modules of BatchNorm and Dropout.
                with MonkeyPatch(pl_module, "log", lambda *args, **kwargs: None):
                    with training_mode(
                        pl_module,
                        excludes=["torch.nn.modules.dropout", "torch.nn.modules.batchnorm"],
                    ):
                        pl_module_step_fn = getattr(pl_module, self.module_step_fn)
                        outputs = pl_module_step_fn(batch, dataloader_idx)
            return outputs

        # Canonicalize the batch to work with Adversary.
        input, target = self.batch_c15n(batch)

        adversary.fit(input, target, model=model)
        input_adv, target_adv = adversary(input, target)

        # Revert to the original batch format.
        batch_adv = self.batch_c15n.revert(input_adv, target_adv)

        return batch_adv
