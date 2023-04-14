#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from typing import TYPE_CHECKING

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

        self.initializer = initializer
        self.composer = composer
        self.gradient_modifier = gradient_modifier or GradientModifier()
        self.projector = projector or Projector()

        self.perturbation = None

    def configure_perturbation(self, input: torch.Tensor | tuple):
        def create_and_initialize(inp):
            pert = torch.empty_like(inp, dtype=torch.float, requires_grad=True)
            self.initializer(pert)
            return pert

        if isinstance(input, tuple):
            self.perturbation = tuple(create_and_initialize(inp) for inp in input)
        elif isinstance(input, torch.Tensor):
            self.perturbation = create_and_initialize(input)
        else:
            raise NotImplementedError

    def parameters(self):
        if self.perturbation is None:
            raise MisconfigurationException("You need to call configure_perturbation before fit.")

        params = self.perturbation
        if not isinstance(params, tuple):
            # FIXME: Should we treat the batch dimension as independent parameters?
            params = (params,)

        return params

    def forward(self, **batch):
        if self.perturbation is None:
            raise MisconfigurationException(
                "You need to call the configure_perturbation before forward."
            )

        # Always perturb the current input.
        self.projector(self.perturbation, **batch)
        input_adv = self.composer(self.perturbation, **batch)

        return input_adv
