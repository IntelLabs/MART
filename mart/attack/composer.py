#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import abc
from typing import Any

import torch

__all__ = ["ModalityComposer"]


class Composer(abc.ABC):
    def __call__(
        self,
        perturbation: torch.Tensor,
        *,
        input: torch.Tensor,
        target: torch.Tensor | dict[str, Any],
        **kwargs,
    ) -> torch.Tensor:
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


class ModalityComposer(Composer):
    def __init__(self, **modality_composers):
        self.modality_composers = modality_composers

    def __call__(
        self,
        perturbation: torch.Tensor | tuple,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        modality: str = "default",
        **kwargs,
    ) -> torch.Tensor | tuple:
        """Recursively compose output from perturbation and input."""
        if isinstance(perturbation, torch.Tensor):
            # Finally we can compose output with tensors.
            composer = self.modality_composers[modality]
            output = composer(perturbation, input=input, target=target)
            return output
        elif isinstance(perturbation, dict):
            # The dict input has modalities specified in keys, passing them recursively.
            output = {}
            for modality in perturbation.keys():
                output[modality] = self(
                    perturbation[modality], input=input[modality], target=target, modality=modality
                )
            return output
        elif isinstance(perturbation, (list, tuple)):
            # The list or tuple input is a collection of sub-input and sub-target.
            output = []
            for pert_i, input_i, target_i in zip(perturbation, input, target):
                output.append(self(pert_i, input=input_i, target=target_i))
            if isinstance(perturbation, tuple):
                output = tuple(output)
            return output

    # We have to implement an abstract method...
    def compose(
        self,
        perturbation: torch.Tensor,
        *,
        input: torch.Tensor,
        target: torch.Tensor | dict[str, Any],
    ) -> torch.Tensor:
        pass
