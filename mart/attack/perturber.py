#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from .projector import Projector

if TYPE_CHECKING:
    from .composer import Composer
    from .initializer import Initializer

__all__ = ["Perturber"]


class Perturber(torch.nn.Module):
    def __init__(
        self,
        *,
        initializer: Initializer,
        composer: Composer,
        projector: Projector | None = None,
        size: Iterable[int] | None = None,
    ):
        """_summary_

        Args:
            initializer (Initializer): To initialize the perturbation.
            composer (Composer): A module which composes adversarial input from input and perturbation.
            projector (Projector): To project the perturbation into some space.
        """
        super().__init__()

        self.initializer_ = initializer
        self.composer = composer
        self.projector = projector or Projector()

        self.perturbation = None

        # FIXME: Should this be in UniversalAdversary?
        if size is not None:
            self.configure_perturbation(torch.empty(size))

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

    def parameters(self):
        if self.perturbation is None:
            raise MisconfigurationException("You need to call configure_perturbation before fit.")

        return super().parameters()

    def forward(self, **batch):
        if self.perturbation is None:
            raise MisconfigurationException(
                "You need to call the configure_perturbation before forward."
            )

        self.projector(self.perturbation, **batch)
        input_adv = self.composer(self.perturbation, **batch)

        return input_adv
