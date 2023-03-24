#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from .gradient_modifier import GradientModifier
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

        self.initializer = initializer
        self.optimizer_fn = optimizer
        self.composer = composer
        self.gradient_modifier = gradient_modifier or GradientModifier()
        self.projector = projector or Projector()
        self.gain_output = gain
        self.objective_fn = objective

        self.perturbation = None

    def configure_perturbation(self, input: torch.Tensor | list[torch.Tensor]):
        def create_and_initialize(inp):
            pert = torch.empty_like(inp, dtype=torch.float, requires_grad=True)
            self.initializer(pert)
            return pert

        if not isinstance(input, list):
            self.perturbation = create_and_initialize(input)
        else:
            self.perturbation = [create_and_initialize(inp) for inp in input]

    def configure_optimizers(self):
        if self.perturbation is None:
            raise MisconfigurationException(
                "You need to call the configure_perturbation before fit."
            )

        params = self.perturbation
        if not isinstance(params, list):
            # FIXME: Should we treat the batch dimension as independent parameters?
            params = (params,)

        return self.optimizer_fn(params)

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

    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        # Configuring gradient clipping in pl.Trainer is still useful, so use it.
        super().configure_gradient_clipping(
            optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
        )

        for group in optimizer.param_groups:
            self.gradient_modifier(group["params"])

    def forward(
        self,
        *,
        input: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor | dict[str, Any] | list[Any],
        **kwargs,
    ):
        if self.perturbation is None:
            raise MisconfigurationException(
                "You need to call the configure_perturbation before forward."
            )

        def project_and_compose(pert, inp, tar):
            self.projector(pert, inp, tar)
            return self.composer(pert, input=inp, target=tar)

        if not isinstance(self.perturbation, list):
            input_adv = project_and_compose(self.perturbation, input, target)
        else:
            input_adv = [
                project_and_compose(pert, inp, tar)
                for pert, inp, tar in zip(self.perturbation, input, target)
            ]

        return input_adv
