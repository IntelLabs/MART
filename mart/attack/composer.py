#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import abc
from typing import Any

import torch
from torchvision.transforms import functional as F


class Composer(abc.ABC):
    def __call__(
        self,
        perturbation: torch.Tensor | tuple,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        **kwargs,
    ) -> torch.Tensor | tuple:
        if isinstance(perturbation, tuple):
            input_adv = tuple(
                self.compose(perturbation_i, input=input_i, target=target_i)
                for perturbation_i, input_i, target_i in zip(perturbation, input, target)
            )
        else:
            input_adv = self.compose(perturbation, input=input, target=target)

        return input_adv

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


class RectanglePatchPerspectiveAdditiveMask(Composer):
    def compose(self, perturbation, *, input, target):
        coords = coords = target["patch_coords"]

        # 1. Pad perturbation to the same size of input.
        height, width = input.shape[-2:]
        height_pert, width_pert = perturbation.shape[-2:]
        pad_left = min(coords[0, 0], coords[3, 0])
        pad_top = min(coords[0, 1], coords[1, 1])
        pad_right = width - width_pert - pad_left
        pad_bottom = height - height_pert - pad_top

        perturbation_padded = F.pad(
            img=perturbation,
            padding=[pad_left, pad_top, pad_right, pad_bottom],
            fill=0,
            padding_mode="constant",
        )

        # 2. Perspective transformation: rectangle -> coords.
        top_left = [pad_left, pad_top]
        top_right = [width - pad_right, pad_top]
        bottom_right = [width - pad_right, height - pad_bottom]
        bottom_left = [pad_left, height - pad_bottom]
        startpoints = [top_left, top_right, bottom_right, bottom_left]
        endpoints = coords
        perturbation_transformed = F.perspective(
            img=perturbation_padded,
            startpoints=startpoints,
            endpoints=endpoints,
            interpolation=F.InterpolationMode.BILINEAR,
            fill=0,
        )

        # 3. Mask.
        mask = target["perturbable_mask"]
        perturbation_masked = perturbation_transformed * mask

        # 4. Addition.
        input_adv = input + perturbation_masked

        # 5. Clamping.
        input_adv = input_adv.clamp(min=0, max=255)
        return input_adv
