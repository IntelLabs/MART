#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import pytorch_lightning as pl
import torch

from .gradient_modifier import GradientModifier
from .perturbation_manager import PerturbationManager
from .projector import Projector

if TYPE_CHECKING:
    from .composer import Composer
    from .initializer import Initializer
    from .objective import Objective

__all__ = ["Perturber"]


class Perturber(pl.LightningModule):
    """Peturbation optimization module."""

    def __init__(
        self,
        *,
        initializer: Initializer,
        optimizer: Callable,
        composer: Composer,
        gradient_modifier: GradientModifier | None = None,
        projector: Projector | None = None,
        gain: str = "loss",
        objective: Objective | None = None,
        optim_params: dict | None = None,
    ):
        """_summary_

        Args:
            initializer (Initializer): To initialize the perturbation.
            optimizer (torch.optim.Optimizer): A PyTorch optimizer.
            composer (Composer): A module which composes adversarial input from input and perturbation.
            gradient_modifier (GradientModifier): To modify the gradient of perturbation.
            projector (Projector): To project the perturbation into some space.
            gain (str): Which output to use as an adversarial gain function, which is a differentiable estimate of adversarial objective. (default: loss)
            objective (Objective): A function for computing adversarial objective, which returns True or False. Optional.
        """
        super().__init__()

        self.optimizer_fn = optimizer
        self.composer = composer
        self.gain_output = gain
        self.objective_fn = objective

        # An object manage the perturbation in both the tensor and the parameter form.
        # FIXME: gradient_modifier should be a hook operating on .grad directly.
        self.pert_manager = PerturbationManager(
            initializer=initializer,
            gradient_modifier=gradient_modifier,
            projector=projector,
            optim_params=optim_params,
        )

    def initialize(self, *, input, **kwargs):
        self.pert_manager.initialize(input)

    def project(self, input, target):
        return self.pert_manager.project(input, target)

    @property
    def perturbation(self):
        return self.pert_manager.perturbation

    def configure_optimizers(self):
        # Parameter initialization is done in Adversary before fit() by invoking initialize(input).
        param_groups = self.pert_manager.parameter_groups()
        return self.optimizer_fn(param_groups)

    def training_step(self, batch, batch_idx):
        # copy batch since we modify it and it is used internally
        batch = batch.copy()

        # We need to evaluate the perturbation against the whole model, so call it normally to get a gain.
        model = batch.pop("model")
        outputs = model(**batch)

        # FIXME: This should really be just `return outputs`. But this might require a new sequence?
        # FIXME: Everything below here should live in the model as modules.
        gain = outputs[self.gain_output]

        # objective_fn is optional, because adversaries may never reach their objective.
        if self.objective_fn is not None:
            found = self.objective_fn(**outputs)

            # No need to calculate new gradients if adversarial examples are already found.
            if len(gain.shape) > 0:
                gain = gain[~found]

        if len(gain.shape) > 0:
            gain = gain.sum()

        return gain

    def forward(
        self,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        **kwargs,
    ):
        # Project perturbation...
        self.project(input, target)

        # Compose adversarial input.
        input_adv = self.composer(self.perturbation, input=input, target=target)

        return input_adv
