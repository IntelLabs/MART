#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import abc
import random
from typing import Any, Iterable

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


class Composite(Composer):
    """We assume an adversary underlays a patch to the input."""

    def __init__(self, premultiplied_alpha=False, use_masks=False):
        super().__init__()

        self.premultiplied_alpha = premultiplied_alpha
        self.use_masks = use_masks

    def compose(self, perturbation, *, input, target):
        # True is mutable, False is immutable.
        perturbable_mask = target["perturbable_mask"]

        # Convert mask to a Tensor with same torch.dtype and torch.device as input,
        #   because some data modules (e.g. Armory) gives binary mask.
        perturbable_mask = perturbable_mask.to(input)

        # Zero out perturbation and mask that overlaps any objects
        if self.use_masks:
            foreground_mask = target["masks"]
            perturbation = perturbation * (1 - foreground_mask)
            perturbable_mask = perturbable_mask * (1 - foreground_mask)

        if not self.premultiplied_alpha:
            perturbation = perturbation * perturbable_mask

        return input * (1 - perturbable_mask) + perturbation


class MaskAdditive(Composer):
    """We assume an adversary adds masked perturbation to the input."""

    def compose(self, perturbation, *, input, target):
        mask = target["perturbable_mask"]
        masked_perturbation = perturbation * mask

        return input + masked_perturbation


# FIXME: It would be really nice if we could compose composers just like we can compose everything else...
class WarpComposite(Composite):
    def __init__(
        self,
        warp,
        *args,
        clamp=(0, 255),
        premultiplied_alpha=True,
        **kwargs,
    ):
        super().__init__(*args, premultiplied_alpha=premultiplied_alpha, **kwargs)

        self.warp = warp
        self.clamp = clamp

    def compose(self, perturbation, *, input, target):
        crop = T.RandomCrop(input.shape[-2:], pad_if_needed=True)

        # Create mask of ones to keep track of filled in pixels
        mask = torch.ones_like(perturbation[:1])

        # Add mask to perturbation so we can keep track of warping. Note the use of
        # premultiplied alpha here.
        mask_perturbation = torch.cat((mask, mask * perturbation))

        # Apply warp transform and crop/pad to input size
        mask_perturbation = self.warp(mask_perturbation)
        mask_perturbation = crop(mask_perturbation)

        # Clamp perturbation to input min/max
        perturbation = mask_perturbation[1:]
        perturbation.clamp_(*self.clamp)

        # Set mask for super().compose
        target["perturbable_mask"] = mask_perturbation[:1]

        return super().compose(perturbation, input=input, target=target)


# FIXME: It would be really nice if we could compose composers just like we can compose everything else...
class ColorJitterWarpComposite(WarpComposite):
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
