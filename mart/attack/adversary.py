#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from itertools import repeat
from typing import TYPE_CHECKING, Any, Callable

import pytorch_lightning as pl
import torch

from mart.utils import silent

if TYPE_CHECKING:
    from .gradient_modifier import GradientModifier
    from .initializer import Initializer
    from .objective import Objective
    from .projector import Projector
    from .threat_model import ThreatModel

__all__ = ["Adversary", "LitPerturber"]


class Adversary(torch.nn.Module):
    def __init__(
        self,
        *,
        max_iters: int = 10,
        trainer: Trainer | None = None,
        perturber: LitPerturber | None = None,
        **kwargs,
    ):
        """_summary_

        Args:
            max_iters (int): The max number of attack iterations.
            trainer (Trainer): A PyTorch-Lightning Trainer object used to fit the perturber.
            perturber (LitPerturber): A LitPerturber that manages perturbations.
        """
        super().__init__()

        self.max_iters = max_iters

        # FIXME: Setup logging directory correctly
        self.attacker = trainer
        if self.attacker is None:
            self.attacker = pl.Trainer(
                accelerator="auto",
                num_sanity_val_steps=0,
                log_every_n_steps=1,
                max_epochs=1,
                callbacks=list(kwargs.pop("callbacks", {}).values()),
                enable_model_summary=False,
                enable_checkpointing=False,
            )

        self.perturber = perturber
        if self.perturber is None:
            self.perturber = LitPerturber(**kwargs)

    @silent()
    def forward(self, **batch):
        if "model" in batch:
            # Initialize perturbation because fit will call configure_optimizers and
            # we want a fresh perturbation.
            self.perturber.initialize_parameters(**batch)

            # Repeat batch max_iters times
            attack_dataloader = repeat(batch, self.max_iters)

            # Attack for an epoch
            self.attacker.fit(model=self.perturber, train_dataloaders=attack_dataloader)

            # Enable future attacks to fit by increasing max_epochs
            self.attacker.fit_loop.max_epochs += 1

        return self.perturber(**batch)


class LitPerturber(pl.LightningModule):
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

        self.perturbation = None

    def configure_optimizers(self):
        assert self.perturbation is not None

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

        # FIXME: Make objective a part of the model...
        # objective_fn is optional, because adversaries may never reach their objective.
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
        if self.perturbation is None:
            self.initialize_parameters(input=input, target=target, **kwargs)

        # Projected perturbation...
        if self.projector is not None:
            self.projector(self.perturbation.data, input, target)

        # ...and apply threat model.
        return self.threat_model(input, target, self.perturbation)

    def initialize_parameters(
        self,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        **kwargs,
    ):
        # Create new perturbation, if necessary
        if self.perturbation is None or self.perturbation.shape != input.shape:
            self.perturbation = torch.empty_like(input, requires_grad=True)

        # FIXME: initializer should really take input and return a perturbation.
        #        once this is done I think this function can just take kwargs?
        self.initializer(self.perturbation)

        # FIXME: I think it's better to use a PL hook here
        if self.gradient_modifier is not None:
            self.perturbation.register_hook(self.gradient_modifier)
