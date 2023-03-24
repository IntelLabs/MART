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

        if not isinstance(initializer, dict):
            initializer = {None: initializer}

        if gradient_modifier is None and not isinstance(gradient_modifier, dict):
            gradient_modifier = {None: gradient_modifier}

        if projector is not None and not isinstance(projector, dict):
            projector = {None: projector}

        self.initializer = initializer
        self.gradient_modifier = gradient_modifier
        self.projector = projector

        if optim_params is None:
            optim_params = {None: {}} | {modality: {} for modality in self.initializer.keys()}
        self.optim_params = optim_params

        self._perturbation = None

    @property
    def perturbation(self):
        # Return perturbation that is homomorphic with input, even if the underlying perturbation could be its sub-components.
        return self._perturbation

    def initialize(self, input):
        # Create raw perturbation that is used to produce parameter groups for optimization.
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
            # TODO: self.gradient_modifier[modality](pert)
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
            self.projector[modality](perturbation, input, target)
        elif isinstance(input, dict):
            for modality_i, input_i in input.items():
                self._project(perturbation[modality_i], input_i, target, modality=modality_i)
        elif isinstance(input, list) or isinstance(input, tuple):
            for perturbation_i, input_i, target_i in zip(perturbation, input, target):
                self._project(perturbation_i, input_i, target_i, modality=modality)

    def parameter_groups(self):
        param_groups = self._parameter_groups(self._perturbation)
        return param_groups

    def _parameter_groups(self, pert, modality=None):
        """Return parameter groups as a list of dictionaries."""

        if isinstance(pert, torch.Tensor):
            return [{"params": pert} | self.optim_params[modality]]
        elif isinstance(pert, dict):
            ret = [self._parameter_groups(pert_i, modality) for modality, pert_i in pert.items()]
            return list(itertools.chain.from_iterable(ret))
        elif isinstance(pert, list) or isinstance(pert, tuple):
            param_list = []
            for pert_i in pert:
                param_list.extend(self._parameter_groups(pert_i))
            return param_list

    def __call__(self, input):
        self.initialize(input)
        return self.perturbation
