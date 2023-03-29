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
        # FIXME: input can be a dictionary {"rgb": tensor}
        mask = mask.to(input)

        return input * (1 - mask) + perturbation * mask


class ModalityComposer(Composer):
    def __init__(self, **modality_composers):
        self.modality_composers = modality_composers

    def _compose(self, perturbation, *, input, target, modality="default"):
        """Recursively compose output from perturbation and input."""
        if isinstance(perturbation, torch.Tensor):
            composer = self.modality_composers[modality]
            output = composer(perturbation, input=input, target=target)
            return output
        elif isinstance(perturbation, dict):
            output = {}
            for modality, pert in perturbation.items():
                output[modality] = self._compose(
                    pert, input=input[modality], target=target, modality=modality
                )
            return output
        elif isinstance(perturbation, list) or isinstance(perturbation, tuple):
            output = []
            for pert_i, input_i, target_i in zip(perturbation, input, target):
                output.append(self._compose(pert_i, input=input_i, target=target_i))
            if isinstance(perturbation, tuple):
                output = tuple(output)
            return output

    def __call__(
        self,
        perturbation: torch.Tensor | tuple,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        **kwargs,
    ) -> torch.Tensor | tuple:
        output = self._compose(perturbation, input=input, target=target)
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
