#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from pytorch_lightning import LightningModule, Trainer

from .callbacks import Callback
from .gain import Gain
from .objective import Objective
from .perturber import BatchPerturber, Perturber
from .threat_model import ThreatModel

__all__ = ["Adversary"]


class AdversaryCallbackHookMixin(Callback):
    """Define event hooks in the Adversary Loop for callbacks."""

    callbacks = {}

    def on_run_start(self, **kwargs) -> None:
        """Prepare the attack loop state."""
        for _name, callback in self.callbacks.items():
            # FIXME: Skip incomplete callback instance.
            # Give access of self to callbacks by `adversary=self`.
            callback.on_run_start(**kwargs)

    def on_examine_start(self, **kwargs) -> None:
        for _name, callback in self.callbacks.items():
            callback.on_examine_start(**kwargs)

    def on_examine_end(self, **kwargs) -> None:
        for _name, callback in self.callbacks.items():
            callback.on_examine_end(**kwargs)

    def on_advance_start(self, **kwargs) -> None:
        for _name, callback in self.callbacks.items():
            callback.on_advance_start(**kwargs)

    def on_advance_end(self, **kwargs) -> None:
        for _name, callback in self.callbacks.items():
            callback.on_advance_end(**kwargs)

    def on_run_end(self, **kwargs) -> None:
        for _name, callback in self.callbacks.items():
            callback.on_run_end(**kwargs)


class LitPerturbation(LightningModule):
    """Peturbation optimization module."""

    def __init__(self, *, batch, optimizer, gain, **kwargs):
        """_summary_

        Args:
        """
        super().__init__()

        self.batch = batch
        self.optimizer_fn = optimizer
        self.gain = gain

        # Perturbation will be same size as batch input
        self.perturbation = torch.nn.Parameter(torch.zeros_like(batch["input"], dtype=torch.float))

    def train_dataloader(self):
        from itertools import cycle

        return cycle([self.batch])

    def configure_optimizers(self):
        return self.optimizer_fn(self.parameters())

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        return outputs[self.gain]

    def forward(self, *, input, target, model, step=None, **kwargs):
        # Calling model with model=None will trigger perturbation application
        return model(input=input, target=target, model=None, step=step)


class Adversary(torch.nn.Module):
    """An adversary module which generates and applies perturbation to input."""

    def __init__(
        self,
        *,
        threat_model: ThreatModel,
        perturber: BatchPerturber | Perturber,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        gain: Gain,
        objective: Objective | None = None,
        callbacks: dict[str, Callback] | None = None,
        **kwargs,
    ):
        """_summary_

        Args:
            threat_model (ThreatModel): A layer which injects perturbation to input, serving as the preprocessing layer to the target model.
            perturber (BatchPerturber | Perturber): A module that stores perturbations.
            optimizer (torch.optim.Optimizer): A PyTorch optimizer.
            max_iters (int): The max number of attack iterations.
            gain (Gain): An adversarial gain function, which is a differentiable estimate of adversarial objective.
            objective (Objective | None): A function for computing adversarial objective, which returns True or False. Optional.
            callbacks (dict[str, Callback] | None): A dictionary of callback objects. Optional.
        """
        super().__init__(**kwargs)

        self.threat_model = threat_model
        self.perturber = perturber
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.gain = gain
        self.objective = objective
        self.callbacks = callbacks

    def forward(
        self,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        model: torch.nn.Module | None = None,
        **kwargs,
    ):
        # Generate a perturbation only if we have a model. This will update
        # the parameters of self.perturbation.
        if model is not None:
            batch = {"input": input, "target": target, "model": model, **kwargs}
            self.perturbation = LitPerturbation(
                batch=batch, optimizer=self.optimizer, gain=self.gain, **kwargs
            )

            # FIXME: how do we get a proper device?
            attacker = Trainer(accelerator="auto", max_steps=self.max_iters)
            attacker.fit(model=self.perturbation)

        # Get perturbation and apply threat model
        # The mask projector in perturber may require information from target.
        # FIXME: Generalize this so we can just pass perturbation.parameters() to threat_model
        perturbation = list(self.perturbation.parameters())[0].to(input.device)
        output = self.threat_model(input, target, perturbation)

        return output
