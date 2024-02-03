#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import torch
from torchvision.transforms.functional import to_pil_image

from mart.nn import SequentialDict

if TYPE_CHECKING:
    from ..perturber import Perturber

__all__ = ["Composer", "Additive", "Mask", "Overlay"]


class Composer(torch.nn.Module):
    def __init__(self, perturber: Perturber, modules, sequence, visualize: bool = False) -> None:
        """_summary_

        Args:
            perturber (Perturber): Manage perturbations.
            functions (dict[str, Function]): A dictionary of functions for composing pertured input.
        """
        super().__init__()

        self.perturber = perturber

        # Convert dict sequences to list sequences by sorting keys
        if isinstance(sequence, dict):
            sequence = [sequence[key] for key in sorted(sequence)]
        self.functions = SequentialDict(modules, {"composer": sequence})
        self.visualize = visualize

    def configure_perturbation(self, input: torch.Tensor | Iterable[torch.Tensor]):
        return self.perturber.configure_perturbation(input)

    def forward(
        self,
        *,
        input: torch.Tensor | Iterable[torch.Tensor],
        target: torch.Tensor | Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor | Iterable[torch.Tensor]:
        perturbation = self.perturber(input=input, target=target)

        if isinstance(perturbation, torch.Tensor) and isinstance(input, torch.Tensor):
            return self._compose(perturbation, input=input, target=target)

        elif (
            isinstance(perturbation, Iterable)
            and isinstance(input, Iterable)  # noqa: W503
            and isinstance(target, Iterable)  # noqa: W503
        ):
            # FIXME: replace tuple with whatever input's type is
            return tuple(
                self._compose(perturbation_i, input=input_i, target=target_i)
                for perturbation_i, input_i, target_i in zip(perturbation, input, target)
            )

        else:
            raise NotImplementedError

    def _compose(
        self,
        perturbation: torch.Tensor,
        *,
        input: torch.Tensor,
        target: torch.Tensor | dict[str, Any],
    ) -> torch.Tensor:
        # A computational graph in SequentialDict().
        output = self.functions(
            input=input, target=target, perturbation=perturbation, step="composer"
        )

        # Visualize intermediate images.
        if self.visualize:
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    to_pil_image(value / 255).save(f"{key}.png")

        # SequentialDict returns a dictionary DotDict,
        #  but we only need the return value of the most recently executed module.
        last_added_key = next(reversed(output))
        output = output[last_added_key]

        # Return the composed input.
        return output


class Additive(torch.nn.Module):
    """We assume an adversary adds perturbation to the input."""

    def forward(self, perturbation, input):
        input = input + perturbation
        return input


class Mask(torch.nn.Module):
    def forward(self, perturbation, mask):
        perturbation = perturbation * mask
        return perturbation


class Overlay(torch.nn.Module):
    """We assume an adversary overlays a patch to the input."""

    def forward(self, perturbation, input, mask):
        # True is mutable, False is immutable.
        # Convert mask to a Tensor with same torch.dtype and torch.device as input,
        #   because some data modules (e.g. Armory) gives binary mask.
        mask = mask.to(input)

        perturbation = perturbation * mask

        input = input * (1 - mask) + perturbation
        return input
