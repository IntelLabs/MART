#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import abc
from typing import Any

import torch

__all__ = ["ModalityEnforcer"]


class ConstraintViolated(Exception):
    pass


class Constraint(abc.ABC):
    @abc.abstractclassmethod
    def __call__(self, input_adv, *, input, target) -> None:
        raise NotImplementedError


class Range(Constraint):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, input_adv, *, input, target):
        if torch.any(input_adv < self.min) or torch.any(input_adv > self.max):
            raise ConstraintViolated(f"Adversarial input is outside [{self.min}, {self.max}].")


class Lp(Constraint):
    def __init__(self, eps: float, p: int | float | None = torch.inf, dim=None, keepdim=False):
        self.p = p
        self.eps = eps
        self.dim = dim
        self.keepdim = keepdim

    def __call__(self, input_adv, *, input, target):
        perturbation = input_adv - input
        norm_vals = perturbation.norm(p=self.p, dim=self.dim, keepdim=self.keepdim)
        norm_max = norm_vals.max()
        if norm_max > self.eps:
            raise ConstraintViolated(
                f"L-{self.p} norm of perturbation exceeds {self.eps}, reaching {norm_max}"
            )


class Integer(Constraint):
    def __init__(self, rtol=0, atol=0, equal_nan=False):
        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan

    def __call__(self, input_adv, *, input, target):
        if not torch.isclose(
            input_adv, input_adv.round(), rtol=self.rtol, atol=self.atol, equal_nan=self.equal_nan
        ).all():
            raise ConstraintViolated("The adversarial example is not in the integer domain.")


class Mask(Constraint):
    def __call__(self, input_adv, *, input, target):
        # True/1 is mutable, False/0 is immutable.
        # mask.shape=(H, W)
        mask = target["perturbable_mask"]

        # Immutable boolean mask, True is immutable.
        imt_mask = (1 - mask).bool()
        perturbation = input_adv - input
        if perturbation.masked_select(imt_mask).any():
            raise ConstraintViolated("Perturbable mask is violated.")


class Enforcer:
    def __init__(self, constraints=None) -> None:
        self.constraints = constraints or {}

    def _check_constraints(self, input_adv, *, input, target):
        for constraint in self.constraints.values():
            constraint(input_adv, input=input, target=target)

    @torch.no_grad()
    def __call__(self, input_adv, *, input, target, **kwargs):
        self._check_constraints(input_adv, input=input, target=target)


class ModalityEnforcer(Enforcer):
    def __init__(self, sub_enforcers=None) -> None:
        self.sub_enforcers = sub_enforcers

    def _enforce(self, input_adv, *, input, target, modality=None):
        if isinstance(input_adv, torch.Tensor):
            self.sub_enforcers[modality](input_adv, input=input, target=target)
        elif isinstance(input_adv, dict):
            for modality in input_adv:
                self._enforce(
                    input_adv[modality], input=input[modality], target=target, modality=modality
                )
        elif isinstance(input_adv, list) or isinstance(input_adv, tuple):
            for input_adv_i, input_i, target_i in zip(input_adv, input, target):
                self._enforce(input_adv_i, input=input_i, target=target_i)

    def __call__(self, input_adv, *, input, target, **kwargs):
        self._enforce(input_adv, input=input, target=target)


class BatchEnforcer(Enforcer):
    @torch.no_grad()
    def __call__(
        self,
        input_adv: torch.Tensor | tuple,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        **kwargs,
    ) -> torch.Tensor | tuple:
        for input_adv_i, input_i, target_i in zip(input_adv, input, target):
            # FIXME: Make it modality-aware.
            self._check_constraints(input_adv_i["rgb"], input=input_i["rgb"], target=target_i)
