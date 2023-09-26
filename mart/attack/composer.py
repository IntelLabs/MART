#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import abc
from collections import OrderedDict
from typing import Any, Iterable

import torch
from torchvision.transforms import functional as F


class Function(torch.nn.Module):
    def __init__(self, *args, order=0, **kwargs) -> None:
        """A stackable function for Composer.

        Args:
            order (int, optional): The priority number. A smaller number makes a function run earlier than others in a sequence. Defaults to 0.
        """
        super().__init__(*args, **kwargs)
        self.order = order

    @abc.abstractmethod
    def forward(self, perturbation, input, target) -> None:
        """Returns the modified perturbation, modified input and target, so we can chain Functions
        in a Composer."""
        pass


class Composer:
    def __init__(self, functions: dict[str, Function]) -> None:
        # Sort functions by function.order and the name.
        self.functions_dict = OrderedDict(
            sorted(functions.items(), key=lambda name_fn: (name_fn[1].order, name_fn[0]))
        )
        self.functions = list(self.functions_dict.values())

    def __call__(
        self,
        perturbation: torch.Tensor | Iterable[torch.Tensor],
        *,
        input: torch.Tensor | Iterable[torch.Tensor],
        target: torch.Tensor | Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor | Iterable[torch.Tensor]:
        if isinstance(perturbation, torch.Tensor) and isinstance(input, torch.Tensor):
            return self.compose(perturbation, input=input, target=target)

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

    def compose(
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


class RectangleCrop(Function):
    def __init__(self, *args, coords_key="patch_coords", **kwargs):
        super().__init__(*args, **kwargs)
        self.coords_key = coords_key

    def get_smallest_rectangle_shape(self, input, patch_coords):
        """Get a smallest rectangle that covers the whole patch."""
        coords = patch_coords
        leading_dims = list(input.shape[:-2])
        width = coords[:, 0].max() - coords[:, 0].min()
        height = coords[:, 1].max() - coords[:, 1].min()
        shape = list(leading_dims) + [height, width]
        return shape

    def slice_rectangle(self, perturbation, height_patch, width_patch):
        """Slice a rectangle from top-left of the perturbation."""
        height_patch_index = torch.tensor(range(height_patch), device=perturbation.device)
        width_patch_index = torch.tensor(range(width_patch), device=perturbation.device)
        perturbation_patch = perturbation.index_select(-2, height_patch_index).index_select(
            -1, width_patch_index
        )
        return perturbation_patch

    def forward(self, perturbation, input, target):
        coords = target[self.coords_key]
        # TODO: Make composers stackable to reuse some Composer.
        # The perturbation variable has the same shape as input.
        #    We slice a small rectangle from top-left of the perturbation variable to compose the patch.
        rectangle_shape = self.get_smallest_rectangle_shape(input, coords)
        # Assume perturbation is in shape of [N]CHW
        height_patch, width_patch = rectangle_shape[-2:]
        rectangle_patch = self.slice_rectangle(perturbation, height_patch, width_patch)
        return rectangle_patch, input, target


class RectanglePad(Function):
    def __init__(self, *args, coords_key="patch_coords", rect_coords_key="rect_coords", **kwargs):
        super().__init__(*args, **kwargs)
        self.coords_key = coords_key
        self.rect_coords_key = rect_coords_key

    def forward(self, perturbation_patch, input, target):
        coords = target[self.coords_key]
        height, width = input.shape[-2:]
        # Pad rectangle to the same size of input, so that it is almost aligned with the patch.
        height_patch, width_patch = perturbation_patch.shape[-2:]
        pad_left = min(coords[0, 0], coords[3, 0])
        pad_top = min(coords[0, 1], coords[1, 1])
        pad_right = width - width_patch - pad_left
        pad_bottom = height - height_patch - pad_top

        perturbation_padded = F.pad(
            img=perturbation_patch,
            padding=[pad_left, pad_top, pad_right, pad_bottom],
            fill=0,
            padding_mode="constant",
        )

        # Save coords of four corners of the rectangle for later transform.
        top_left = [pad_left, pad_top]
        top_right = [width - pad_right, pad_top]
        bottom_right = [width - pad_right, height - pad_bottom]
        bottom_left = [pad_left, height - pad_bottom]
        target[self.rect_coords_key] = [top_left, top_right, bottom_right, bottom_left]

        return perturbation_padded, input, target


class RectanglePerspectiveTransform(Function):
    def __init__(self, *args, coords_key="patch_coords", rect_coords_key="rect_coords", **kwargs):
        super().__init__(*args, **kwargs)
        self.coords_key = coords_key
        self.rect_coords_key = rect_coords_key

    def forward(self, perturbation_rect, input, target):
        coords = target[self.coords_key]
        # Perspective transformation: rectangle -> coords.
        # Fetch four corners of the rectangle.
        startpoints = target[self.rect_coords_key]
        endpoints = coords
        # TODO: Make interpolation configurable.
        perturbation_coords = F.perspective(
            img=perturbation_rect,
            startpoints=startpoints,
            endpoints=endpoints,
            interpolation=F.InterpolationMode.BILINEAR,
            fill=0,
        )
        return perturbation_coords, input, target


class FakeClamp(Function):
    """A Clamp operation that preserves gradients."""

    def __init__(self, *args, min_val, max_val, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_val = min_val
        self.max_val = max_val

    @staticmethod
    def fake_clamp(x, *, min_val, max_val):
        with torch.no_grad():
            x_clamped = x.clamp(min_val, max_val)
            diff = x_clamped - x
        return x + diff

    def forward(self, perturbation, input, target):
        input = self.fake_clamp(input, min_val=self.min_val, max_val=self.max_val)
        return perturbation, input, target
