#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Sequence

import torch
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from .projector import Projector

if TYPE_CHECKING:
    from .initializer import Initializer

__all__ = ["Perturber", "UniversalPerturber"]


class Perturber(torch.nn.Module):
    def __init__(
        self,
        *,
        initializer: Initializer,
        projector: Projector | None = None,
        grad_modifier: Callable = None,
    ):
        """_summary_

        Args:
            initializer (Initializer): To initialize the perturbation.
            projector (Projector): To project the perturbation into some space.
            grad_modifier (Callable): A non in-place operation being hooked by Tensor.register_hook().
        """
        super().__init__()

        self.initializer_ = initializer
        self.projector_ = projector or Projector()
        self.grad_modifier = grad_modifier
        # We can call the handler to remove the hook later by self._grad_modifier_hook_handler.remove()
        self._grad_modifier_hook_handler = None

        self.perturbation = None

    def configure_perturbation(self, input: torch.Tensor | Iterable[torch.Tensor]):
        def matches(input, perturbation):
            if perturbation is None:
                return False

            if isinstance(input, torch.Tensor) and isinstance(perturbation, torch.Tensor):
                return input.shape == perturbation.shape

            if isinstance(input, Iterable) and isinstance(perturbation, Iterable):
                if len(input) != len(perturbation):
                    return False

                return all(
                    [
                        matches(input_i, perturbation_i)
                        for input_i, perturbation_i in zip(input, perturbation)
                    ]
                )

            return False

        def create_from_tensor(tensor):
            if isinstance(tensor, torch.Tensor):
                return torch.nn.Parameter(
                    torch.empty_like(tensor, dtype=torch.float, requires_grad=True)
                )
            elif isinstance(tensor, Iterable):
                return torch.nn.ParameterList([create_from_tensor(t) for t in tensor])
            else:
                raise NotImplementedError

        # If we have never created a perturbation before or perturbation does not match input, then
        # create a new perturbation.
        if not matches(input, self.perturbation):
            self.perturbation = create_from_tensor(input)

        # Always (re)initialize perturbation.
        self.initializer_(self.perturbation)

    def named_parameters(self, *args, **kwargs):
        if self.perturbation is None:
            raise MisconfigurationException("You need to call configure_perturbation before fit.")

        return super().named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        if self.perturbation is None:
            raise MisconfigurationException("You need to call configure_perturbation before fit.")

        return super().parameters(*args, **kwargs)

    def forward(self, **batch):
        if self.perturbation is None:
            raise MisconfigurationException(
                "You need to call the configure_perturbation before forward."
            )

        self.projector_(self.perturbation, **batch)
        # We need to register the hook at every forward pass.
        if self.grad_modifier:
            self._grad_modifier_hook_handler = self.perturbation.register_hook(self.grad_modifier)

        return self.perturbation


class UniversalPerturber(Perturber):
    """The perturbation is shared by all data batches, so we don't duplicate the shape of any
    batch."""

    def __init__(
        self,
        *,
        shape: Sequence[int],
        initializer: Initializer,
        projector: Projector | None = None,
        grad_modifier: Callable = None,
    ):
        super().__init__(initializer=initializer, projector=projector, grad_modifier=grad_modifier)

        # We just configure the perturbation here. No need to invoke configure_perturbation() externally.
        self._configure_perturbation(shape)

    def configure_perturbation(self, input: torch.Tensor | Iterable[torch.Tensor]):
        raise NotImplementedError(
            "We don't (repeatedly) configure the universal perturbation with input."
        )

    def _configure_perturbation(self, shape: Sequence[int]):
        perturbation = torch.empty(shape, dtype=torch.float, requires_grad=True)
        self.perturbation = torch.nn.Parameter(perturbation)
        # Always initialize the perturbation.
        self.initializer_(self.perturbation)
