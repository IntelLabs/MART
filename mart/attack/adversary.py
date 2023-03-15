#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import logging
from collections import OrderedDict
from itertools import repeat
from typing import TYPE_CHECKING, Any, Callable

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch.nn.modules.lazy import LazyModuleMixin

if TYPE_CHECKING:
    from .gradient_modifier import GradientModifier
    from .initializer import Initializer
    from .objective import Objective
    from .projector import Projector
    from .threat_model import ThreatModel

__all__ = ["Adversary"]


class Adversary(torch.nn.Module):
    def __init__(
        self,
        *,
        max_iters: int = 10,
        callbacks: dict[str, Callback] | None = None,
        **kwargs,
    ):
        """_summary_

        Args:
            max_iters (int): The max number of attack iterations.
            callbacks (dict[str, Callback] | None): A dictionary of callback objects. Optional.
        """
        super().__init__()

        self.max_iters = max_iters
        self.callbacks = callbacks

        self.perturber = LitPerturber(**kwargs)

        # FIXME: Setup logging directory correctly
        self.attacker = Trainer(
            accelerator="auto",
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            max_epochs=1,
            enable_model_summary=False,
            callbacks=list(self.callbacks.values()),  # ignore keys
            enable_checkpointing=False,
        )

    def forward(self, **batch):
        if "model" in batch:
            self.perturber.initialize_parameters(**batch)

            # Repeat batch max_iters times
            attack_dataloader = repeat(batch, self.max_iters)

            with Silence():
                # Attack for another epoch
                self.attacker.fit_loop.max_epochs += 1
                self.attacker.fit(model=self.perturber, train_dataloaders=attack_dataloader)

        return self.perturber(**batch)


class LitPerturber(LightningModule):
    """Peturbation optimization module."""

    def __init__(
        self,
        *,
        initializer: Initializer,
        optimizer: Callable,
        threat_model: ThreatModel,
        gradient_modifier: GradientModifier | None = None,
        projector: Projector | None = None,
        gain: str = "loss",
        objective: Objective | None = None,
    ):
        """_summary_

        Args:
            initializer (Initializer): To initialize the perturbation.
            optimizer (torch.optim.Optimizer): A PyTorch optimizer.
            threat_model (ThreatModel): A layer which injects perturbation to input, serving as the preprocessing layer to the target model.
            gradient_modifier (GradientModifier): To modify the gradient of perturbation.
            projector (Projector): To project the perturbation into some space.
            gain (str): Which output to use as an adversarial gain function, which is a differentiable estimate of adversarial objective. (default: loss)
            objective (Objective): A function for computing adversarial objective, which returns True or False. Optional.
        """
        super().__init__()

        self.initializer = initializer
        self.optimizer_fn = optimizer
        self.threat_model = threat_model
        self.gradient_modifier = gradient_modifier
        self.projector = projector
        self.gain_output = gain
        self.objective_fn = objective
        self.projector = projector

        self.perturbation = None

    def configure_optimizers(self):
        return self.optimizer_fn([self.perturbation])

    def training_step(self, batch, batch_idx):
        # copy batch since we modify it and it is used internally
        batch = batch.copy()
        input = batch.pop("input")
        target = batch.pop("target")
        model = batch.pop("model")

        outputs = model(input=input, target=target, **batch)
        # FIXME: This should really be just `return outputs`. Everything below here should live in the model!
        gain = outputs[self.gain_output]

        # objective_fn is optional, because adversaries may never reach their objective.
        # FIXME: Make objective a part of the model...
        if self.objective_fn is not None:
            found = self.objective_fn(**outputs)
            self.log("found", found.sum().float(), prog_bar=True)

            # No need to calculate new gradients if adversarial examples are already found.
            if len(gain.shape) > 0:
                gain = gain[~found]

        if len(gain.shape) > 0:
            gain = gain.sum()

        self.log("gain", gain, prog_bar=True)

        return gain

    def forward(
        self,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        **kwargs,
    ):
        # Get projected perturbation and apply threat model
        # The mask projector in perturber may require information from target.
        self.projector(self.perturbation.data, input, target)
        return self.threat_model(input, target, self.perturbation)

    def initialize_parameters(
        self,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        **kwargs,
    ):
        # Create new perturbation if necessary
        if self.perturbation is None or self.perturbation.shape != input.shape:
            self.perturbation = torch.zeros_like(input, requires_grad=True)

        # FIXME: initialize should really take input and return a perturbation...
        #        once this is done I think this function can just take kwargs?
        self.initializer(self.perturbation)

        if self.gradient_modifier is not None:
            self.perturbation.register_hook(self.gradient_modifier)

        # Keep perturbation on input device since fit moves it to CPU
        return self.perturbation.to(input.device)


class Silence:
    """Suppress logging."""

    DEFAULT_NAMES = ["pytorch_lightning.utilities.rank_zero", "pytorch_lightning.accelerators.gpu"]

    def __init__(self, names=None):
        if names is None:
            names = Silence.DEFAULT_NAMES

        self.loggers = [logging.getLogger(name) for name in names]

    def __enter__(self):
        for logger in self.loggers:
            logger.propagate = False

    def __exit__(self, exc_type, exc_value, traceback):
        for logger in self.loggers:
            logger.propagate = False
