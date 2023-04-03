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

    MODALITY_DEFAULT = "default"

    def __init__(
        self,
        *,
        optimizer: Callable,
        gain: Gain,
        composer: Composer | dict[str, Composer],
        initializer: Initializer | dict[str, Initializer],
        gradient_modifier: GradientModifier | dict[str, GradientModifier] | None = None,
        projector: Projector | dict[str, Projector] | None = None,
        objective: Objective | None = None,
        optim_params: dict[str, dict[str, Any]] | None = None,
    ):
        """_summary_

        Args:
            optimizer: A partial of PyTorch optimizer that awaits parameters to optimize.
            gain: An adversarial gain function, which is a differentiable estimate of adversarial objective.
            composer: A module which composes adversarial input from input and perturbation. Modality-aware.
            initializer: To initialize the perturbation. Modality-aware.
            gradient_modifier: To modify the gradient of perturbation. Modality-aware.
            projector: To project the perturbation into some space. Modality-aware.
            objective: A function for computing adversarial objective, which returns True or False. Optional.
            optim_params: A dictionary of optimization hyper-parameters. E.g. {"rgb": {"lr": 0.1}}.
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
            initializer = {self.MODALITY_DEFAULT: initializer}
        if not isinstance(gradient_modifier, dict):
            gradient_modifier = {self.MODALITY_DEFAULT: gradient_modifier}
        if not isinstance(projector, dict):
            projector = {self.MODALITY_DEFAULT: projector}
        if not isinstance(composer, dict):
            composer = {self.MODALITY_DEFAULT: composer}

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
        def create_and_initialize(data, *, input, target, modality):
            # Though data and target are not used, they are required placeholders for modality_dispatch().
            # TODO: we don't want an integer tensor, but make sure it does not affect mixed precision training.
            pert = torch.empty_like(input, dtype=torch.float, requires_grad=True)
            self.initializer[modality](pert)
            return pert

        # Recursively configure perturbation in tensor.
        # Though only input=input is used, we have to fill the placeholders of data and target.
        self.perturbation = modality_dispatch(
            create_and_initialize, input, input=input, target=input, modality=self.MODALITY_DEFAULT
        )

    def parameter_groups(self):
        """Extract parameter groups for optimization from perturbation tensor(s)."""
        param_groups = self._parameter_groups(self.perturbation, modality=self.MODALITY_DEFAULT)
        return param_groups

    def _parameter_groups(self, pert, *, modality):
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

    def project_(self, perturbation, *, input, target, **kwargs):
        """In-place projection."""
        modality_dispatch(
            self.projector,
            perturbation,
            input=input,
            target=target,
            modality=self.MODALITY_DEFAULT,
        )

    def compose(self, perturbation, *, input, target, **kwargs):
        return modality_dispatch(
            self.composer, perturbation, input=input, target=target, modality=self.MODALITY_DEFAULT
        )

    def configure_optimizers(self):
        # parameter_groups is generated from perturbation.
        if self.perturbation is None:
            raise MisconfigurationException(
                "You need to call the configure_perturbation before fit."
            )
        return self.optimizer_fn(self.parameter_groups())

    def training_step(self, batch, batch_idx):
        # copy batch since we modify it and it is used internally
        batch = batch.copy()

        # We need to evaluate the perturbation against the whole model, so call it normally to get a gain.
        model = batch.pop("model")
        # When an Adversary takes input from another module in the sequence, we would have to specify kwargs of Adversary, and model would be a required kwarg.
        outputs = model(**batch, model=None)

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
            modality = self.MODALITY_DEFAULT if "modality" not in group else group["modality"]
            self.gradient_modifier[modality](group["params"])

    def forward(self, **batch):
        if self.perturbation is None:
            raise MisconfigurationException(
                "You need to call the configure_perturbation before forward."
            )

        self.project_(self.perturbation, **batch)
        input_adv = self.compose(self.perturbation, **batch)

        return input_adv
