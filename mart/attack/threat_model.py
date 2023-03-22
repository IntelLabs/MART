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


class Constraint(abc.ABC):
    @abc.abstractclassmethod
    def __call__(self, input_adv, input, target) -> None:
        raise NotImplementedError


class Range(Constraint):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, input_adv, input, target):
        if torch.any(input_adv < self.min) or torch.any(input_adv > self.max):
            raise ValueError(f"Adversarial input is outside [{self.min}, {self.max}].")


class Lp(Constraint):
    def __init__(self, eps: float, p: int | float | None = torch.inf, dim=None, keepdim=False):
        self.p = p
        self.eps = eps
        self.dim = dim
        self.keepdim = keepdim

    def __call__(self, input_adv, input, target):
        perturbation = input_adv - input
        norm_vals = perturbation.norm(p=self.p, dim=self.dim, keepdim=self.keepdim)
        norm_max = norm_vals.max()
        if norm_max > self.eps:
            raise ValueError(
                f"L-{self.p} norm of perturbation exceeds {self.eps}, reaching {norm_max}"
            )


class Integer(Constraint):
    def __init__(self, rtol=0, atol=0, equal_nan=False):
        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan

    def __call__(self, input_adv, input, target):
        if not torch.isclose(
            input_adv, input_adv.round(), rtol=self.rtol, atol=self.atol, equal_nan=self.equal_nan
        ).all():
            raise ValueError("The adversarial example is not in the integer domain.")


class Mask(Constraint):
    def __call__(self, input_adv, input, target):
        # True/1 is mutable, False/0 is immutable.
        # mask.shape=(H, W)
        mask = target["perturbable_mask"]

        # Immutable boolean mask, True is immutable.
        imt_mask = (1 - mask).bool()
        perturbation = input_adv - input
        if perturbation.masked_select(imt_mask).any():
            raise ValueError("Perturbable mask is violated.")


class Enforcer(torch.nn.Module):
    def __init__(self, constraints=None) -> None:
        super().__init__()
        self.constraints = constraints or {}

    def _check_constraints(self, input_adv, input, target):
        for constraint in self.constraints.values():
            constraint(input_adv, input, target)

    @torch.no_grad()
    def forward(self, input_adv, input, target):
        self._check_constraints(input_adv, input, target)


class BatchEnforcer(Enforcer):
    @torch.no_grad()
    def forward(
        self,
        input_adv: torch.Tensor | tuple,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
    ) -> torch.Tensor | tuple:
        for input_adv_i, input_i, target_i in zip(input_adv, input, target):
            self._check_constraints(input_adv_i, input_i, target_i)


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
