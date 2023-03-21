#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import abc
from typing import Any, Dict, Optional, Union

import torch

__all__ = ["BatchComposer"]


class Constraint(abc.ABC):
    @abc.abstractclassmethod
    def __call__(self, input, target, input_adv) -> None:
        raise NotImplementedError


class Range(Constraint):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, input, target, input_adv):
        perturbation = input_adv - input
        if torch.any(perturbation < self.min) or torch.any(perturbation > self.max):
            raise ValueError(f"Perturbation is outside [{self.min}, {self.max}].")


class Lp(Constraint):
    def __init__(
        self, eps: float, p: Optional[Union[int, float]] = torch.inf, dim=None, keepdim=False
    ):
        self.p = p
        self.eps = eps
        self.dim = dim
        self.keepdim = keepdim

    def __call__(self, input, target, input_adv):
        perturbation = input_adv - input
        norm_vals = perturbation.norm(p=self.p, dim=self.dim, keepdim=self.keepdim)
        norm_max = norm_vals.max()
        if norm_max > self.eps:
            raise ValueError(
                f"L-{self.p} norm of perturbation exceeds {self.eps}, reaching {norm_max}"
            )


class Integer(Constraint):
    def __call__(self, input, target, input_adv):
        torch.testing.assert_close(input_adv, input_adv.round())


class Mask(Constraint):
    def __call__(self, input, target, input_adv):
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
        self.constraints = constraints

    def _check_constraints(self, input, target, input_adv) -> Any:
        if self.constraints is not None:
            for constraint in self.constraints.values():
                constraint(input, target, input_adv)

    @torch.no_grad()
    def forward(self, input, target, input_adv) -> Any:
        self._check_constraints(input, target, input_adv)


class BatchEnforcer(Enforcer):
    @torch.no_grad()
    def forward(
        self,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        input_adv: Union[torch.Tensor, tuple],
    ) -> Union[torch.Tensor, tuple]:
        for input_i, target_i, input_adv_i in zip(input, target, input_adv):
            self._check_constraints(input_i, target_i, input_adv_i)


class Composer(torch.nn.Module, abc.ABC):
    pass


class BatchComposer(Composer):
    def __init__(self, composer: Composer):
        super().__init__()

        self.composer = composer

    def forward(
        self,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        perturbation: Union[torch.Tensor, tuple],
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
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
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        perturbation: Union[torch.Tensor, tuple],
    ) -> Union[torch.Tensor, tuple]:
        return input + perturbation


class Overlay(Composer):
    """We assume an adversary overlays a patch to the input."""

    def forward(
        self,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        perturbation: Union[torch.Tensor, tuple],
    ) -> Union[torch.Tensor, tuple]:

        # True is mutable, False is immutable.
        mask = target["perturbable_mask"]

        # Convert mask to a Tensor with same torch.dtype and torch.device as input,
        #   because some data modules (e.g. Armory) gives binary mask.
        mask = mask.to(input)

        return input * (1 - mask) + perturbation * mask
