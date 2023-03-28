#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import itertools
from typing import Callable

import torch


class PerturbationManager:
    def __init__(
        self,
        *,
        initializer: Callable | dict,
        gradient_modifier: Callable | dict | None = None,
        projector: Callable | dict | None = None,
        optim_params: dict | None = None,
    ) -> None:

        # In case gradient_modifier or projector is None.
        def nop(*args, **kwargs):
            pass

        gradient_modifier = gradient_modifier or nop
        projector = projector or nop

        # Backward compatibility, in case modality is unknown, and not given in input.
        if not isinstance(initializer, dict):
            initializer = {None: initializer}
        if not isinstance(gradient_modifier, dict):
            gradient_modifier = {None: gradient_modifier}
        if not isinstance(projector, dict):
            projector = {None: projector}

        # In case optimization parameters are not given.
        optim_params = optim_params or {modality: {} for modality in initializer.keys()}

        self.initializer = initializer
        self.gradient_modifier = gradient_modifier
        self.projector = projector
        self.optim_params = optim_params

        self._perturbation = None

    @property
    def perturbation(self):
        """Return perturbation that is homomorphic with input."""
        # TODO: Compose perturbation from sub-componenets.
        return self._perturbation

    def initialize(self, input):
        """Create and initialize raw perturbation components.

        With raw perturbation components, we can
            1. compose perturbation that is homomorphic to input.
            2. compose parameter groups for optimization.
        """
        # TODO: Raw perturbation is not necessarily homorphic with input.
        self._perturbation = self._initialize(input)

    def _initialize(self, input, modality=None):
        """Recursively materialize and initialize perturbation that is homomorphic as input; Hook
        gradient modifiers."""
        if isinstance(input, torch.Tensor):
            # Materialize.
            pert = torch.zeros_like(input, requires_grad=True)

            # Initialize.
            self.initializer[modality](pert)

            # Gradient modifier hook.
            # FIXME: use actual gradient modifier, self.gradient_modifier[modality](pert)
            #        The current implementation of gradient modifiers is not hookable.
            if self.gradient_modifier is not None:
                pert.register_hook(lambda grad: grad.sign())

            return pert
        elif isinstance(input, dict):
            return {modality: self._initialize(inp, modality) for modality, inp in input.items()}
        elif isinstance(input, list):
            return [self._initialize(inp) for inp in input]
        elif isinstance(input, tuple):
            return tuple(self._initialize(inp) for inp in input)

    def project(self, input, target):
        if self.projector is not None:
            self._project(self._perturbation, input, target)

    def _project(self, perturbation, input, target, modality=None):
        """Recursively project perturbation tensors that may hide behind dictionaries, list or
        tuple."""
        if isinstance(input, torch.Tensor):
            self.projector[modality](perturbation, input=input, target=target)
        elif isinstance(input, dict):
            for modality_i, input_i in input.items():
                self._project(perturbation[modality_i], input_i, target, modality=modality_i)
        elif isinstance(input, list) or isinstance(input, tuple):
            for perturbation_i, input_i, target_i in zip(perturbation, input, target):
                self._project(perturbation_i, input_i, target_i, modality=modality)

    @property
    def parameter_groups(self):
        param_groups = self._parameter_groups(self._perturbation)
        return param_groups

    def _parameter_groups(self, pert, modality=None):
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

    def __call__(self, input):
        self.initialize(input)
        return self.perturbation
