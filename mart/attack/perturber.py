#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import itertools
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

        self.optimizer_fn = optimizer
        self.composer = composer
        self.gain_fn = gain
        self.objective_fn = objective

        # FIXME: gradient_modifier should be a hook operating on .grad directly.

        # In case gradient_modifier or projector is None.
        def nop(*args, **kwargs):
            pass

        gradient_modifier = gradient_modifier or nop
        projector = projector or nop

        # Backward compatibility, in case modality is unknown, and not given in input.
        if not isinstance(initializer, dict):
            initializer = {"default": initializer}
        if not isinstance(gradient_modifier, dict):
            gradient_modifier = {"default": gradient_modifier}
        if not isinstance(projector, dict):
            projector = {"default": projector}

        # In case optimization parameters are not given.
        optim_params = optim_params or {modality: {} for modality in initializer.keys()}

        self.initializer = initializer
        self.gradient_modifier = gradient_modifier
        self.projector = projector
        self.optim_params = optim_params

        self.perturbation = None

    def configure_perturbation(self, input):
        # Recursively configure perturbation in tensor.
        self.perturbation = self._configure_perturbation(input)

    def _configure_perturbation(self, input, modality="default"):
        """Recursively create and initialize perturbation that is homomorphic as input; Hook
        gradient modifiers."""
        if isinstance(input, torch.Tensor):
            # Create.
            pert = torch.empty_like(input, requires_grad=True)

            # Initialize.
            self.initializer[modality](pert)

            # Gradient modifier hook.
            # FIXME: use actual gradient modifier, self.gradient_modifier[modality](pert)
            #        The current implementation of gradient modifiers is not hookable.
            if self.gradient_modifier is not None:
                pert.register_hook(lambda grad: grad.sign())

            return pert
        elif isinstance(input, dict):
            return {
                modality: self._configure_perturbation(inp, modality)
                for modality, inp in input.items()
            }
        elif isinstance(input, list):
            return [self._configure_perturbation(inp) for inp in input]
        elif isinstance(input, tuple):
            return tuple(self._configure_perturbation(inp) for inp in input)

    @property
    def parameter_groups(self):
        """Extract parameter groups for optimization from perturbation tensor(s)."""
        param_groups = self._parameter_groups(self.perturbation)
        return param_groups

    def _parameter_groups(self, pert, modality="default"):
        """Recursively return parameter groups as a list of dictionaries."""

        if isinstance(pert, torch.Tensor):
            return [{"params": pert} | self.optim_params[modality]]
        elif isinstance(pert, dict):
            ret = [self._parameter_groups(pert_i, modality) for modality, pert_i in pert.items()]
            # Concatenate a list of lists.
            return list(itertools.chain.from_iterable(ret))
        elif isinstance(pert, list) or isinstance(pert, tuple):
            param_list = []
            for pert_i in pert:
                param_list.extend(self._parameter_groups(pert_i))
            return param_list

    def project(self, input, target):
        if self.projector is not None:
            self._project(self.perturbation, input=input, target=target)

    def _project(self, perturbation, *, input, target, modality="default"):
        """Recursively project perturbation tensors that may hide behind dictionaries, list or
        tuple."""
        if isinstance(input, torch.Tensor):
            self.projector[modality](perturbation, input=input, target=target)
        elif isinstance(input, dict):
            for modality_i, input_i in input.items():
                self._project(
                    perturbation[modality_i], input=input_i, target=target, modality=modality_i
                )
        elif isinstance(input, list) or isinstance(input, tuple):
            for perturbation_i, input_i, target_i in zip(perturbation, input, target):
                self._project(perturbation_i, input=input_i, target=target_i, modality=modality)

    def configure_optimizers(self):
        # Parameter initialization is done in Adversary before fit() by invoking initialize(input).
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

    def forward(
        self,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        **kwargs,
    ):
        if self.perturbation is None:
            raise MisconfigurationException(
                "You need to call the configure_perturbation before forward."
            )

        # Project perturbation...
        self.project(input, target)

        # Compose adversarial input.
        input_adv = self.composer(self.perturbation, input=input, target=target)

        return input_adv
