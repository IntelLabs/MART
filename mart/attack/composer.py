#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import abc
from typing import Any, Iterable

import torch


class Composer(abc.ABC):
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
