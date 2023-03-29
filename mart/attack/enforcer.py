#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import abc
from typing import Any

import torch

__all__ = ["Enforcer"]


class ConstraintViolated(Exception):
    pass


class Constraint(abc.ABC):
    def __call__(
        self,
        input_adv: torch.Tensor,
        *,
        input: torch.Tensor,
        target: torch.Tensor | dict[str, Any],
    ) -> None:
        self.verify(input_adv, input=input, target=target)

    @abc.abstractmethod
    def verify(
        self,
        input_adv: torch.Tensor,
        *,
        input: torch.Tensor,
        target: torch.Tensor | dict[str, Any],
    ) -> None:
        raise NotImplementedError


class Range(Constraint):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def verify(self, input_adv, *, input, target):
        if torch.any(input_adv < self.min) or torch.any(input_adv > self.max):
            raise ConstraintViolated(f"Adversarial input is outside [{self.min}, {self.max}].")


class Lp(Constraint):
    def __init__(
        self, eps: float, p: int | float = torch.inf, dim: int | None = None, keepdim: bool = False
    ):
        self.p = p
        self.eps = eps
        self.dim = dim
        self.keepdim = keepdim

    def verify(self, input_adv, *, input, target):
        perturbation = input_adv - input
        norm_vals = perturbation.norm(p=self.p, dim=self.dim, keepdim=self.keepdim)
        norm_max = norm_vals.max()
        if norm_max > self.eps:
            raise ConstraintViolated(
                f"L-{self.p} norm of perturbation exceeds {self.eps}, reaching {norm_max}"
            )


class Integer(Constraint):
    def __init__(self, rtol: float = 0.0, atol: float = 0.0, equal_nan: bool = False):
        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan

    def verify(self, input_adv, *, input, target):
        if not torch.isclose(
            input_adv, input_adv.round(), rtol=self.rtol, atol=self.atol, equal_nan=self.equal_nan
        ).all():
            raise ConstraintViolated("The adversarial example is not in the integer domain.")


class Mask(Constraint):
    def verify(self, input_adv, *, input, target):
        # True/1 is mutable, False/0 is immutable.
        # mask.shape=(H, W)
        mask = target["perturbable_mask"]

        # Immutable boolean mask, True is immutable.
        imt_mask = (1 - mask).bool()
        perturbation = input_adv - input
        if perturbation.masked_select(imt_mask).any():
            raise ConstraintViolated("Perturbable mask is violated.")


class Enforcer:
    def __init__(self, **modality_constraints: dict[str, dict[str, Constraint]]) -> None:
        self.modality_constraints = modality_constraints

    @torch.no_grad()
    def _enforce(
        self,
        input_adv: torch.Tensor,
        *,
        input: torch.Tensor,
        target: torch.Tensor | dict[str, Any],
        modality: str,
    ):
        for constraint in self.modality_constraints[modality].values():
            constraint(input_adv, input=input, target=target)

    def __call__(
        self,
        input_adv: torch.Tensor | tuple | list[torch.Tensor] | dict[str, torch.Tensor],
        *,
        input: torch.Tensor | tuple | list[torch.Tensor] | dict[str, torch.Tensor],
        target: torch.Tensor | dict[str, Any],
        modality: str = "constraints",
        **kwargs,
    ):
        if isinstance(input_adv, torch.Tensor):
            # Finally we can verify constraints on tensor, per its modality.
            # Set modality="constraints" by default, so that it is backward compatible with existing configs without modalities.
            self._enforce(input_adv, input=input, target=target, modality=modality)
        elif isinstance(input_adv, dict):
            # The dict input has modalities specified in keys, passing them recursively.
            for modality in input_adv:
                self(input_adv[modality], input=input[modality], target=target, modality=modality)
        elif isinstance(input_adv, (list, tuple)):
            # The list or tuple input is a collection of sub-input.
            for input_adv_i, input_i, target_i in zip(input_adv, input, target):
                self(input_adv_i, input=input_i, target=target_i)
