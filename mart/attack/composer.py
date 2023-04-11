#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import abc
from typing import Any

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


class Composer(abc.ABC):
    def __call__(
        self,
        perturbation: torch.Tensor | tuple,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        **kwargs,
    ) -> torch.Tensor | tuple:
        if isinstance(perturbation, tuple) and isinstance(input, tuple):
            return tuple(
                self.compose(perturbation_i, input=input_i, target=target_i)
                for perturbation_i, input_i, target_i in zip(perturbation, input, target)
            )

        elif isinstance(perturbation, torch.Tensor) and isinstance(input, tuple):
            return tuple(
                self.compose(perturbation, input=input_i, target=target_i)
                for input_i, target_i in zip(input, target)
            )

        elif isinstance(perturbation, torch.Tensor) and isinstance(input, torch.Tensor):
            return self.compose(perturbation, input=input, target=target)

        else:
            raise NotImplementedError

    @abc.abstractmethod
    def compose(
        self,
        perturbation: torch.Tensor,
        *,
        input: torch.Tensor,
        target: torch.Tensor | dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError


class Additive(Composer):
    """We assume an adversary adds perturbation to the input."""

    def compose(self, perturbation, *, input, target):
        return input + perturbation


class Overlay(Composer):
    """We assume an adversary overlays a patch to the input."""

    def compose(self, perturbation, *, input, target):
        # True is mutable, False is immutable.
        mask = target["perturbable_mask"]

        # Convert mask to a Tensor with same torch.dtype and torch.device as input,
        #   because some data modules (e.g. Armory) gives binary mask.
        mask = mask.to(input)

        return input * (1 - mask) + perturbation * mask


class MaskAdditive(Composer):
    """We assume an adversary adds masked perturbation to the input."""

    def compose(self, perturbation, *, input, target):
        mask = target["perturbable_mask"]
        masked_perturbation = perturbation * mask

        return input + masked_perturbation


class RandomAffineOverlay(Overlay):
    def __init__(
        self,
        degrees,
        translate=None,
        scale=None,
        shear=None,
        clamp=(0, 255),
    ):
        self.transform = T.RandomAffine(
            degrees,
            translate,
            scale,
            shear=shear,
            #interpolation=InterpolationMode.BILINEAR,
        )
        self.clamp = clamp

    def compose(self, perturbation, *, input, target):
        random_crop = T.RandomCrop(input.shape[-2:], pad_if_needed=True)

        # Create mask of ones to keep track of filled in pixels
        mask = torch.ones_like(perturbation[:1])
        mask_perturbation = torch.cat((mask, perturbation))

        # Apply random affine transform and crop/pad to input size
        mask_perturbation = self.transform(mask_perturbation)
        mask_perturbation = random_crop(mask_perturbation)

        # Clamp perturbation to input min/max
        perturbation = mask_perturbation[1:]
        perturbation.clamp_(*self.clamp)

        # Make mask binary
        mask = mask_perturbation[:1] > 0 # fill=0
        target["perturbable_mask"] = mask

        return super().compose(perturbation, input=input, target=target)
