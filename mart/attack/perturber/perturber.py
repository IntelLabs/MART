#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from __future__ import annotations

from typing import Any

import torch

from ..callbacks import Callback

__all__ = ["Perturber"]


class Perturber(Callback, torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
    """The base class of perturbers.

    A perturber wraps a nn.Parameter and returns this parameter when called. It also enables one to
    specify an initialization for this parameter, how to modify gradients computed on this
    parameter, and how to project the values of the parameter.
    """

    def __init__(
        self,
        initializer: Initializer,
        gradient_modifier: GradientModifier | None = None,
        projector: Projector | None = None,
        **optim_params,
    ):
        """_summary_

        Args:
            initializer (Initializer): To initialize the perturbation.
            gradient_modifier (GradientModifier | None): To modify the gradient of perturbation.
            projector (Projector | None): To project the perturbation into some space.
            optim_params (dict | None): Optimization parameters such learning rate and momentum for perturbation.
        """
        super().__init__()

        self.initializer = initializer
        self.gradient_modifier = gradient_modifier
        self.projector = projector
        self.optim_params = optim_params

        # Register perturbation as a non-persistent buffer even though we will optimize it. This is because it is not
        # a parameter of the underlying model but a parameter of the adversary.
        self.register_buffer("perturbation", torch.nn.UninitializedParameter(), persistent=False)

        def projector_wrapper(perturber_module, args):
            if isinstance(perturber_module.perturbation, torch.nn.UninitializedParameter):
                raise ValueError("Perturbation must be initialized")

            input, target = args
            return projector(perturber_module.perturbation.data, input, target)

        # Will be called before forward() is called.
        if projector is not None:
            self.register_forward_pre_hook(projector_wrapper)

    def parameter_groups(self):
        """Return parameters along with the pre-defined optimization parameters.

        Example: `[{"params": perturbation, "lr":0.1, "momentum": 0.9}]`
        """
        if "params" in self.optim_params:
            raise ValueError(
                'Optimization parameters should not include "params" which will override the actual parameters to be optimized. '
            )

        return [{"params": self.perturbation} | self.optim_params]

    def initialize_parameters(
        self, input: torch.Tensor, target: torch.Tensor | dict[str, Any]
    ) -> None:
        assert isinstance(self.perturbation, torch.nn.UninitializedParameter)

        self.perturbation.materialize(input.shape, device=input.device)

        # A backward hook that will be called when a gradient w.r.t the Tensor is computed.
        if self.gradient_modifier is not None:
            self.perturbation.register_hook(self.gradient_modifier)

        self.initializer(self.perturbation)

    def on_run_start(
        self,
        *,
        adversary: Adversary,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        model: torch.nn.Module,
        **kwargs,
    ):
        # Initialize lazy module
        self(input, target)

    def forward(
        self, input: torch.Tensor, target: torch.Tensor | dict[str, Any]
    ) -> torch.Tensor:
        return self.perturbation

    def extra_repr(self):
        perturbation = self.perturbation
        if not self.has_uninitialized_params():
            perturbation = (perturbation.shape, perturbation.min(), perturbation.max())

        return (
            f"{repr(perturbation)}, initializer={self.initializer},"
            f"gradient_modifier={self.gradient_modifier}, projector={self.projector}"
        )
