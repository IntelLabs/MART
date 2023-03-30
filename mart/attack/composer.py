#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import abc
from typing import Any

import torch

__all__ = ["Composer"]


class Method(abc.ABC):
    """Composition method base class."""

    @abc.abstractmethod
    def __call__(
        self,
        perturbation: torch.Tensor,
        *,
        input: torch.Tensor,
        target: torch.Tensor | dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError


class Additive(Method):
    """We assume an adversary adds perturbation to the input."""

    def __call__(self, perturbation, *, input, target):
        return input + perturbation


class Overlay(Method):
    """We assume an adversary overlays a patch to the input."""

    def __call__(self, perturbation, *, input, target):
        # True is mutable, False is immutable.
        mask = target["perturbable_mask"]

        # Convert mask to a Tensor with same torch.dtype and torch.device as input,
        #   because some data modules (e.g. Armory) gives binary mask.
        mask = mask.to(input)

        return input * (1 - mask) + perturbation * mask


class Composer:
    """A modality-aware composer.

    Example usage: `Composer(rgb=Overlay(), depth=Additive())`. Non-modality composer can be
    defined as `Composer(method=Additive())`.
    """

    def __init__(self, **modality_methods):
        self.modality_methods = modality_methods

    def __call__(
        self,
        perturbation: torch.Tensor | dict[str, torch.Tensor] | tuple | list,
        *,
        input: torch.Tensor | dict[str, torch.Tensor] | tuple | list,
        target: torch.Tensor | dict[str, Any] | tuple | list,
        modality: str = "method",
    ) -> torch.Tensor:
        """Recursively compose output from perturbation and input."""
        assert type(perturbation) == type(input)

        if isinstance(perturbation, torch.Tensor):
            # Finally we can compose output with tensors.
            method = self.modality_methods[modality]
            output = method(perturbation, input=input, target=target)
            return output
        elif isinstance(perturbation, dict):
            # The dict input has modalities specified in keys, passing them recursively.
            # For non-modality input that does not have dict, modality="method" by default.
            output = {}
            for modality in perturbation.keys():
                output[modality] = self(
                    perturbation[modality], input=input[modality], target=target, modality=modality
                )
            return output
        elif isinstance(perturbation, (list, tuple)):
            # We assume a modality-dictionary only contains tensors, but not list/tuple.
            assert modality == "method"
            # The list or tuple input is a collection of sub-input and sub-target.
            output = []
            for pert_i, input_i, target_i in zip(perturbation, input, target):
                output.append(self(pert_i, input=input_i, target=target_i, modality=modality))
            if isinstance(perturbation, tuple):
                output = tuple(output)
            return output
        else:
            raise ValueError(f"Unsupported data type of perturbation: {type(perturbation)}.")
