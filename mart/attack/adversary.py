#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from functools import partial
from itertools import cycle
from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
import torch

from mart.utils import silent

from .perturber import Perturber

if TYPE_CHECKING:
    from .enforcer import Enforcer

__all__ = ["Adversary"]


class Adversary(torch.nn.Module):
    """An adversary module which generates and applies perturbation to input."""

    def __init__(
        self,
        *,
        enforcer: Enforcer,
        perturber: Perturber,
        attacker: pl.Trainer | None = None,
        **kwargs,
    ):
        """_summary_

        Args:
            enforcer (Enforcer): A Callable that enforce constraints on the adversarial input.
            perturber (Perturber): A Perturber that manages perturbations.
            attacker (Trainer): A PyTorch-Lightning Trainer object used to fit the perturber.
        """
        super().__init__()

        self._attacker = attacker

        if self._attacker is None:
            # Enable attack to be late bound in forward
            self._attacker = partial(
                pl.Trainer,
                num_sanity_val_steps=0,
                logger=False,
                max_epochs=0,
                limit_train_batches=kwargs.pop("max_iters", 10),
                callbacks=list(kwargs.pop("callbacks", {}).values()),  # dict to list of values
                enable_model_summary=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

        else:
            # We feed the same batch to the attack every time so we treat each step as an
            # attack iteration. As such, attackers must only run for 1 epoch and must limit
            # the number of attack steps via limit_train_batches.
            assert self._attacker.max_epochs == 0
            assert self._attacker.limit_train_batches > 0

        self.perturber = perturber
        self.enforcer = enforcer

    @silent()
    def forward(self, **batch):
        # Adversary lives within a sequence of model. To signal the adversary should attack, one
        # must pass a model to attack when calling the adversary. Since we do not know where the
        # Adversary lives inside the model, we also need the remaining sequence to be able to
        # get a loss.
        if "model" in batch and "sequence" in batch:
            self._attack(**batch)

        # Always use perturb the current input.
        input_adv = self.perturber(**batch)

        # Enforce constraints after the attack optimization ends.
        if "model" in batch and "sequence" in batch:
            self.enforcer(input_adv, **batch)

        return input_adv

    def _attack(self, input, **batch):
        batch = {"input": input, **batch}

        # Configure and reset perturber to use batch inputs
        self.perturber.configure_perturbation(input)

        # Attack, aka fit a perturbation, for one epoch by cycling over the same input batch.
        # We use Trainer.limit_train_batches to control the number of attack iterations.
        self.attacker.fit_loop.max_epochs += 1
        self.attacker.fit(self.perturber, train_dataloaders=cycle([batch]))

    @property
    def attacker(self):
        if not isinstance(self._attacker, partial):
            return self._attacker

        # Convert torch.device to PL accelerator
        device = self.perturber.device

        if device.type == "cuda":
            accelerator = "gpu"
            devices = [device.index]
        elif device.type == "cpu":
            accelerator = "cpu"
            devices = None
        else:
            accelerator = device.type
            devices = [device.index]

        self._attacker = self._attacker(accelerator=accelerator, devices=devices)

        return self._attacker
