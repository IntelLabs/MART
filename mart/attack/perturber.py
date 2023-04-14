#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from .gradient_modifier import GradientModifier
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
        gradient_modifier: GradientModifier | None = None,
        projector: Projector | None = None,
    ):
        """_summary_

        Args:
            initializer (Initializer): To initialize the perturbation.
            composer (Composer): A module which composes adversarial input from input and perturbation.
            gradient_modifier (GradientModifier): To modify the gradient of perturbation.
            projector (Projector): To project the perturbation into some space.
        """
        super().__init__()

        self.initializer_ = initializer
        self.composer = composer
        self.gradient_modifier = gradient_modifier or GradientModifier()
        self.projector = projector or Projector()

        self.perturbation = None

    def configure_perturbation(self, input: torch.Tensor | Iterable[torch.Tensor]):
        def create_from_tensor(tensor):
            if isinstance(tensor, torch.Tensor):
                return torch.nn.Parameter(
                    torch.empty_like(tensor, dtype=torch.float, requires_grad=True)
                )
            elif isinstance(tensor, Iterable):
                return torch.nn.ParameterList([create_from_tensor(t) for t in tensor])
            else:
                raise NotImplementedError

            # FIXME: Attach gradient modifier

        # If we have never created a perturbation before, then create it.
        if self.perturbation is None:
            self.perturbation = create_from_tensor(input)

        # FIXME: Check if perturbation is same shape as input

        # (re)Use existing perturbations but initialize them.
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

        # Always perturb the current input.
        self.projector(self.perturbation, **batch)
        input_adv = self.composer(self.perturbation, **batch)

        return input_adv
