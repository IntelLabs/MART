#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import torch

__all__ = ["Perturber"]


class Perturber(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
    """The base class of perturbers.

    A perturber wraps a nn.Parameter and returns this parameter when called. It also enables one to
    specify an initialization for this parameter, how to modify gradients computed on this
    parameter, and how to project the values of the parameter.
    """

    def __init__(self, initializer, gradient_modifier=None, projector=None):
        """_summary_

        Args:
            initializer (object): To initialize the perturbation.
            gradient_modifier (object): To modify the gradient of perturbation.
            projector (object): To project the perturbation into some space.
        """
        super().__init__()

        self.initializer = initializer
        self.gradient_modifier = gradient_modifier
        self.projector = projector

        self.perturbation = torch.nn.UninitializedParameter()

        def projector_wrapper(perturber_module, args):
            if isinstance(perturber_module.perturbation, torch.nn.UninitializedParameter):
                raise ValueError("Perturbation must be initialized")

            input, target = args
            return projector(perturber_module.perturbation.data, input, target)

        # Will be called before forward() is called.
        if projector is not None:
            self.register_forward_pre_hook(projector_wrapper)

    def initialize_parameters(self, input, target):
        assert isinstance(self.perturbation, torch.nn.UninitializedParameter)

        self.perturbation.materialize(input.shape, device=input.device)

        # A backward hook that will be called when a gradient w.r.t the Tensor is computed.
        if self.gradient_modifier is not None:
            self.perturbation.register_hook(self.gradient_modifier)

        self.initializer(self.perturbation)

    def forward(self, input, target):
        return self.perturbation

    def extra_repr(self):
        perturbation = self.perturbation
        if not self.has_uninitialized_params():
            perturbation = (perturbation.shape, perturbation.min(), perturbation.max())

        return (
            f"{repr(perturbation)}, initializer={self.initializer},"
            f"gradient_modifier={self.gradient_modifier}, projector={self.projector}"
        )
