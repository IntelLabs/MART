#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import abc
from typing import Any, Iterable

import random
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode


class Composer(abc.ABC):
    def __call__(
        self,
        perturbation: torch.Tensor | Iterable[torch.Tensor],
        *,
        input: torch.Tensor | Iterable[torch.Tensor],
        target: torch.Tensor | Iterable[torch.Tensor | dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor | Iterable[torch.Tensor]:
        if isinstance(perturbation, torch.Tensor) and isinstance(input, torch.Tensor):
            return self.compose(perturbation, input=input, target=target)

        elif (
            isinstance(perturbation, torch.Tensor)
            and isinstance(input, Iterable)  # noqa: W503
            and isinstance(target, Iterable)  # noqa: W503
        ):
            # FIXME: replace tuple with whatever input's type is
            return tuple(
                self.compose(perturbation, input=input_i, target=target_i)
                for input_i, target_i in zip(input, target)
            )

        elif (
            isinstance(perturbation, Iterable)
            and isinstance(input, Iterable)  # noqa: W503
            and isinstance(target, Iterable)  # noqa: W503
        ):
            # FIXME: replace tuple with whatever input's type is
            return tuple(
                self.compose(perturbation_i, input=input_i, target=target_i)
                for perturbation_i, input_i, target_i in zip(perturbation, input, target)
            )

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


# FIXME: It would be really nice if we could compose composers just like we can compose everything else...
class WarpOverlay(Overlay):
    def __init__(
        self,
        warp,
        clamp=(0, 255),
        drop_p=0.5,
        drop_range=(0.1, 0.9),
    ):
        self.warp = warp
        self.clamp = clamp
        self.p = drop_p
        self.drop_range = drop_range

    def compose(self, perturbation, *, input, target):
        crop = T.RandomCrop(input.shape[-2:], pad_if_needed=True)

        # Create mask of ones to keep track of filled in pixels
        mask = torch.ones_like(perturbation[:1])

        # Apply drop block to mask
        if self.p > torch.rand(1):
            # Select random block size
            block_size = [random.uniform(*self.drop_range),
                          random.uniform(*self.drop_range)]

            # Convert to pixel using mask shape
            if block_size[0] < 1:  # height
                block_size[0] = mask.shape[1]*block_size[0]
            if block_size[1] < 1:  # width
                block_size[1] = mask.shape[2]*block_size[1]

            block_size[0] = int(block_size[0])
            block_size[1] = int(block_size[1])

            # Randomly pad block to perturbation shape
            padding_top = random.randint(0, mask.shape[1] - block_size[0])
            padding_bottom = mask.shape[1] - padding_top - block_size[0]
            padding_left = random.randint(0, mask.shape[2] - block_size[1])
            padding_right = mask.shape[2] - padding_left - block_size[1]

            block = torch.zeros(block_size, device=mask.device)
            block = F.pad(block, (padding_left, padding_top, padding_right, padding_bottom), fill=1.)
            mask = mask * block

        # Add mask to perturbation so we can keep track of warping
        mask_perturbation = torch.cat((mask, perturbation))

        # Apply warp transform and crop/pad to input size
        mask_perturbation = self.warp(mask_perturbation)
        mask_perturbation = crop(mask_perturbation)

        # Clamp perturbation to input min/max
        perturbation = mask_perturbation[1:]
        perturbation.clamp_(*self.clamp)

        # Make mask binary
        mask = mask_perturbation[:1] > 0  # fill=0
        target["perturbable_mask"] = mask

        return super().compose(perturbation, input=input, target=target)


# FIXME: It would be really nice if we could compose composers just like we can compose everything else...
class ColorJitterWarpOverlay(WarpOverlay):
    def __init__(
        self,
        *args,
        brightness=0,
        contrast=0,
        saturation=0,
        hue=0,
        pixel_scale=255,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)
        self.pixel_scale = pixel_scale

    def compose(self, perturbation, *, input, target):
        # ColorJitter and friends assume floating point tensors are between [0, 1]...
        perturbation = self.color_jitter(perturbation / self.pixel_scale) * self.pixel_scale

        return super().compose(perturbation, input=input, target=target)
