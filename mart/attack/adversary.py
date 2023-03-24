#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from functools import partial
from itertools import cycle

import pytorch_lightning as pl
import torch

from mart.utils import silent

from .enforcer import Enforcer
from .perturber import Perturber

__all__ = ["Adversary"]


class Adversary(torch.nn.Module):
    """An adversary module which generates and applies perturbation to input."""

    def __init__(
        self,
        *,
        enforcer: Enforcer,
        perturber: Perturber | None = None,
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

        self.attacker = attacker

        if self.attacker is None:
            # Enable attack to be late bound in forward
            self.attacker = partial(
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
            assert self.attacker.max_epochs == 0
            assert self.attacker.limit_train_batches > 0

        self.perturber = perturber or Perturber(**kwargs)
        self.enforcer = enforcer

    @silent()
    def forward(self, **batch):
        # Adversary lives within a sequence of model. To signal the adversary should attack, one
        # must pass a model to attack when calling the adversary. Since we do not know where the
        # Adversary lives inside the model, we also need the remaining sequence to be able to
        # get a loss.
        if "model" in batch and "sequence" in batch:
            # Late bind attacker on same device as input
            if isinstance(self.attacker, partial):
                inputs = batch["input"]

                if isinstance(inputs, tuple):
                    # FIXME: Make it modality-aware to get a tensor.
                    inputs = inputs[0]["rgb"]

                device = inputs.device

                if device.type == "cuda":
                    accelerator = "gpu"
                    devices = [device.index]

                elif device.type == "cpu":
                    accelerator = "cpu"
                    devices = None

                else:
                    accelerator = device.type
                    devices = [device.index]

                self.attacker = self.attacker(accelerator=accelerator, devices=devices)

            # Attack, aka fit a perturbation, for one epoch by cycling over the same input batch.
            # We use Trainer.limit_train_batches to control the number of attack iterations.
            self.attacker.fit_loop.max_epochs += 1

            # Initialize perturbation with input.
            self.perturber.initialize(**batch)

            self.attacker.fit(self.perturber, train_dataloaders=cycle([batch]))

        # Always use perturb the current input.
        input_adv = self.perturber(**batch)

        if "model" in batch and "sequence" in batch:
            # We only enforce constraints after the attack optimization ends.
            self.enforcer(input_adv, **batch)

        return input_adv
