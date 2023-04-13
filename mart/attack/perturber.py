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
    from .gain import Gain
    from .initializer import Initializer
    from .objective import Objective

__all__ = ["Perturber", "UniversalPerturber"]


class Perturber(pl.LightningModule):
    """Peturbation optimization module."""

    def __init__(
        self,
        *,
        initializer: Initializer,
        optimizer: Callable,
        composer: Composer,
        gain: Gain,
        gradient_modifier: GradientModifier | None = None,
        projector: Projector | None = None,
        objective: Objective | None = None,
    ):
        """_summary_

        Args:
            initializer (Initializer): To initialize the perturbation.
            optimizer (torch.optim.Optimizer): A PyTorch optimizer.
            composer (Composer): A module which composes adversarial input from input and perturbation.
            gain (Gain): An adversarial gain function, which is a differentiable estimate of adversarial objective.
            gradient_modifier (GradientModifier): To modify the gradient of perturbation.
            projector (Projector): To project the perturbation into some space.
            objective (Objective): A function for computing adversarial objective, which returns True or False. Optional.
        """
        super().__init__()

        self.initializer = initializer
        self.optimizer_fn = optimizer
        self.composer = composer
        self.gradient_modifier = gradient_modifier or GradientModifier()
        self.projector = projector or Projector()
        self.gain_fn = gain
        self.objective_fn = objective

        self.perturbation = None

    def configure_perturbation(self, input: torch.Tensor | tuple):
        def create_and_initialize(inp):
            pert = torch.empty_like(inp, dtype=torch.float, requires_grad=True)
            self.initializer(pert)
            return pert

        if isinstance(input, tuple):
            self.perturbation = tuple(create_and_initialize(inp) for inp in input)
        elif isinstance(input, torch.Tensor):
            self.perturbation = create_and_initialize(input)
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        if self.perturbation is None:
            raise MisconfigurationException(
                "You need to call the configure_perturbation before fit."
            )

        params = self.perturbation
        if not isinstance(params, tuple):
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
        # Use CallWith to dispatch **outputs.
        gain = self.gain_fn(**outputs)

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

    def forward(self, **batch):
        if self.perturbation is None:
            raise MisconfigurationException(
                "You need to call the configure_perturbation before forward."
            )

        self.projector(self.perturbation, **batch)
        output = self.composer(self.perturbation, **batch)

        torch.save((self.perturbation.detach().cpu(), output), "/tmp/output.pt")  # nosec: B108

        return output


class UniversalPerturber(Perturber):
    def __init__(self, *args, size: tuple, **kwargs):
        super().__init__(*args, **kwargs)

        self.size = size

    def configure_perturbation(self, input: torch.Tensor | tuple):
        if self.perturbation is not None:
            return

        if isinstance(input, tuple):
            device = input[0].device
        elif isinstance(input, torch.Tensor):
            device = input.device
        else:
            raise NotImplementedError

        self.perturbation = torch.empty(self.size, device=device, requires_grad=True)
        self.initializer(self.perturbation)
