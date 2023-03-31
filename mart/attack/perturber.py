#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from mart.utils import modality_dispatch

from .gradient_modifier import GradientModifier
from .projector import Projector

if TYPE_CHECKING:
    from .composer import Composer
    from .gain import Gain
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
        gain: Gain,
        gradient_modifier: GradientModifier | None = None,
        projector: Projector | None = None,
        objective: Objective | None = None,
        optim_params: dict | None = None,
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

        # Modality-neutral objects.
        self.optimizer_fn = optimizer
        self.gain_fn = gain
        self.objective_fn = objective

        # Replace None with nop().
        gradient_modifier = gradient_modifier or GradientModifier()
        projector = projector or Projector()

        # Modality-specific objects.
        # Backward compatibility, in case modality is unknown, and not given in input.
        if not isinstance(initializer, dict):
            initializer = {"default": initializer}
        if not isinstance(gradient_modifier, dict):
            gradient_modifier = {"default": gradient_modifier}
        if not isinstance(projector, dict):
            projector = {"default": projector}
        if not isinstance(composer, dict):
            composer = {"default": composer}

        # Backward compatibility, in case optimization parameters are not given.
        if optim_params is None:
            optim_params = {modality: {} for modality in initializer.keys()}

        # Modality-specific objects.
        self.initializer = initializer
        self.gradient_modifier = gradient_modifier
        self.projector = projector
        self.composer = composer
        self.optim_params = optim_params

        self.perturbation = None

    def configure_perturbation(self, input: torch.Tensor | tuple | tuple[dict[str, torch.Tensor]]):
        def create_and_initialize(data, *, input, target, modality="default"):
            # Though data and target are not used, they are required placeholders for modality_dispatch().
            pert = torch.empty_like(input, requires_grad=True)
            self.initializer[modality](pert)
            return pert

        # Make a dictionary of modality-function.
        modality_func = {
            modality: partial(create_and_initialize, modality=modality)
            for modality in self.initializer
        }

        # Recursively configure perturbation in tensor.
        # Though only input=input is used, we have to fill the placeholders of data and target.
        self.perturbation = modality_dispatch(
            modality_func, input, input=input, target=input, modality="default"
        )

    @property
    def parameter_groups(self):
        """Extract parameter groups for optimization from perturbation tensor(s)."""
        param_groups = self._parameter_groups(self.perturbation)
        return param_groups

    def _parameter_groups(self, pert, modality="default"):
        """Recursively return parameter groups as a list of dictionaries."""

        if isinstance(pert, torch.Tensor):
            # Return a list of dictionary instead of a dictionary, easier to extend later.
            # Add the modality notation so that we can perform gradient modification later.
            return [{"params": pert, "modality": modality} | self.optim_params[modality]]
        elif isinstance(pert, dict):
            param_list = []
            for modality, pert_i in pert.items():
                ret_modality = self._parameter_groups(pert_i, modality=modality)
                param_list.extend(ret_modality)
            return param_list
        elif isinstance(pert, (list, tuple)):
            param_list = []
            for pert_i in pert:
                ret_i = self._parameter_groups(pert_i, modality=modality)
                param_list.extend(ret_i)
            return param_list
        else:
            raise ValueError(f"Unsupported data type of input: {type(pert)}.")

    def project(self, perturbation, *, input, target, **kwargs):
        modality_dispatch(
            self.projector, perturbation, input=input, target=target, modality="default"
        )

    def compose(self, perturbation, *, input, target, **kwargs):
        return modality_dispatch(
            self.composer, perturbation, input=input, target=target, modality="default"
        )

    def configure_optimizers(self):
        # parameter_groups is generated from perturbation.
        if self.perturbation is None:
            raise MisconfigurationException(
                "You need to call the configure_perturbation before fit."
            )
        return self.optimizer_fn(self.parameter_groups)

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
            modality = "default" if "modality" not in group else group["modality"]
            self.gradient_modifier[modality](group["params"])

    def forward(self, **batch):
        if self.perturbation is None:
            raise MisconfigurationException(
                "You need to call the configure_perturbation before forward."
            )

        self.project(self.perturbation, **batch)
        input_adv = self.compose(self.perturbation, **batch)

        return input_adv
