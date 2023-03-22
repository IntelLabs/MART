#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import abc
from typing import Any

import torch

__all__ = ["BatchComposer"]


class Composer(torch.nn.Module, abc.ABC):
    @abc.abstractclassmethod
    def forward(
        self,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        perturbation: torch.Tensor | tuple,
    ) -> torch.Tensor | tuple:
        raise NotImplementedError


class BatchComposer(Composer):
    def __init__(self, composer: Composer):
        super().__init__()

        self.composer = composer

    def forward(
        self,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        perturbation: torch.Tensor | tuple,
        **kwargs,
    ) -> torch.Tensor | tuple:
        output = []

        for input_i, target_i, perturbation_i in zip(input, target, perturbation):
            output_i = self.composer(input_i, target_i, perturbation_i, **kwargs)
            output.append(output_i)

        if isinstance(input, torch.Tensor):
            output = torch.stack(output)
        else:
            output = tuple(output)

        return output


class Additive(Composer):
    """We assume an adversary adds perturbation to the input."""

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor | dict[str, Any],
        perturbation: torch.Tensor,
    ) -> torch.Tensor:
        return input + perturbation


class Overlay(Composer):
    """We assume an adversary overlays a patch to the input."""

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor | dict[str, Any],
        perturbation: torch.Tensor,
    ) -> torch.Tensor:
        # True is mutable, False is immutable.
        mask = target["perturbable_mask"]

        # Convert mask to a Tensor with same torch.dtype and torch.device as input,
        #   because some data modules (e.g. Armory) gives binary mask.
        mask = mask.to(input)

        return input * (1 - mask) + perturbation * mask
