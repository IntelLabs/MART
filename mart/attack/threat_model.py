#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import abc
from typing import Any, Dict, Optional, Union

import torch

__all__ = ["BatchThreatModel"]


class Constraint(abc.ABC):
    @abc.abstractclassmethod
    def __call__(self, perturbation) -> None:
        raise NotImplementedError


class Range(Constraint):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, perturbation):
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

    def __call__(self, perturbation):
        norm_vals = perturbation.norm(p=self.p, dim=self.dim, keepdim=self.keepdim)
        norm_max = norm_vals.max()
        if norm_max > self.eps:
            raise ValueError(
                f"L-{self.p} norm of perturbation exceeds {self.eps}, reaching {norm_max}"
            )


class Integer(Constraint):
    def __call__(self, perturbation):
        torch.testing.assert_close(perturbation, perturbation.round())


class ThreatModel(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(
        self,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        perturbation: Union[torch.Tensor, tuple],
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        raise NotImplementedError


class BatchThreatModel(ThreatModel):
    def __init__(self, threat_model: ThreatModel):
        super().__init__()

        self.threat_model = threat_model

    def forward(
        self,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        perturbation: Union[torch.Tensor, tuple],
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        output = []

        for input_i, target_i, perturbation_i in zip(input, target, perturbation):
            output_i = self.threat_model(input_i, target_i, perturbation_i, **kwargs)
            output.append(output_i)

        if isinstance(input, torch.Tensor):
            output = torch.stack(output)
        else:
            output = tuple(output)

        return output


class Additive(ThreatModel):
    """We assume an adversary adds perturbation to the input."""

    def __init__(self, constraints=None) -> None:
        super().__init__()
        self.constraints = constraints

    def _check_constraints(self, perturbation):
        if self.constraints is not None:
            for constraint in self.constraints.values():
                constraint(perturbation)

    def forward(
        self,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        perturbation: Union[torch.Tensor, tuple],
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        self._check_constraints(perturbation)

        return input + perturbation


class Overlay(ThreatModel):
    """We assume an adversary overlays a patch to the input."""

    def __init__(self, constraints=None) -> None:
        super().__init__()
        self.constraints = constraints

    def _check_constraints(self, perturbation):
        if self.constraints is not None:
            for constraint in self.constraints.values():
                constraint(perturbation)

    def forward(
        self,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        perturbation: Union[torch.Tensor, tuple],
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        self._check_constraints(perturbation)

        # True is mutable, False is immutable.
        mask = target["perturbable_mask"]

        # Convert mask to a Tensor with same torch.dtype and torch.device as input,
        #   because some data modules (e.g. Armory) gives binary mask.
        mask = mask.to(input)

        return input * (1 - mask) + perturbation * mask
