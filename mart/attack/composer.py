#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import abc
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Iterable

import torch
import torchvision
import torchvision.transforms.functional as F

from mart.utils import pylogger

logger = pylogger.get_pylogger(__name__)

if TYPE_CHECKING:
    from .perturber import Perturber


class Function(torch.nn.Module):
    def __init__(self, *args, order=0, **kwargs) -> None:
        """A stackable function for Composer.

        Args:
            order (int, optional): The priority number. A smaller number makes a function run earlier than others in a sequence. Defaults to 0.
        """
        super().__init__(*args, **kwargs)
        self.order = order

    @abc.abstractmethod
    def forward(
        self, perturbation: torch.Tensor, input: torch.Tensor, target: torch.Tensor | dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | dict]:
        """Returns the modified perturbation, modified input and target, so we can chain Functions
        in a Composer."""
        pass


class Composer(torch.nn.Module):
    def __init__(self, perturber: Perturber, functions: dict[str, Function]) -> None:
        """_summary_

        Args:
            perturber (Perturber): Manage perturbations.
            functions (dict[str, Function]): A dictionary of functions for composing pertured input.
        """
        super().__init__()

        self.perturber = perturber

        # Sort functions by function.order and the name.
        self.functions_dict = OrderedDict(
            sorted(functions.items(), key=lambda name_fn: (name_fn[1].order, name_fn[0]))
        )
        self.functions = list(self.functions_dict.values())

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
        for function in self.functions:
            perturbation, input, target = function(perturbation, input, target)

        # Return the composed input.
        return input


class Additive(Function):
    """We assume an adversary adds perturbation to the input."""

    def forward(self, perturbation, input, target):
        input = input + perturbation
        return perturbation, input, target


class Mask(Function):
    def __init__(self, *args, key="perturbable_mask", **kwargs):
        super().__init__(*args, **kwargs)
        self.key = key

    def forward(self, perturbation, input, target):
        mask = target[self.key]
        perturbation = perturbation * mask
        return perturbation, input, target


class Overlay(Function):
    """We assume an adversary overlays a patch to the input."""

    def __init__(self, *args, key="perturbable_mask", **kwargs):
        super().__init__(*args, **kwargs)
        self.key = key

    def forward(self, perturbation, input, target):
        # True is mutable, False is immutable.
        mask = target[self.key]

        # Convert mask to a Tensor with same torch.dtype and torch.device as input,
        #   because some data modules (e.g. Armory) gives binary mask.
        mask = mask.to(input)

        perturbation = perturbation * mask

        input = input * (1 - mask) + perturbation
        return perturbation, input, target


class Image(Function):
    """Add an image to perturbation."""

    def __init__(self, *args, path: str, scale: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = torchvision.io.read_image(path, torchvision.io.ImageReadMode.RGB) / scale

    def forward(self, perturbation, input, target):
        image = self.image

        if image.shape != perturbation.shape:
            logger.info(f"Resizing image from {image.shape} to {perturbation.shape}...")
            image = F.resize(image, perturbation.shape[1:])

        perturbation = perturbation + image
        return perturbation, input, target
