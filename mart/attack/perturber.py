#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import torch
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from .projector import Projector

if TYPE_CHECKING:
    from .initializer import Initializer

__all__ = ["Perturber"]


class Perturber(torch.nn.Module):
    def __init__(
        self,
        *,
        initializer: Initializer,
        projector: Projector | None = None,
    ):
        """_summary_

        Args:
            initializer (Initializer): To initialize the perturbation.
            projector (Projector): To project the perturbation into some space.
        """
        super().__init__()

        self.initializer_ = initializer
        self.projector_ = projector or Projector()

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

        return self.perturbation
