#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import abc
from typing import Optional, Union

import torch

__all__ = ["Initializer"]


class Initializer(abc.ABC):
    """Initializer base class."""

    def __call__(self, input, perturbation=None) -> torch.Tensor:
        if perturbation is None or perturbation.shape != input.shape:
            perturbation = torch.empty_like(input, requires_grad=True)

        self.initialize(perturbation)

        return perturbation

    @abc.abstractmethod
    def initialize(self, perturbation: torch.Tensor) -> None:
        pass


class Constant(Initializer):
    def __init__(self, constant: Optional[Union[int, float]] = 0):
        self.constant = constant

    def initialize(self, perturbation: torch.Tensor) -> None:
        torch.nn.init.constant_(perturbation, self.constant)


class Uniform(Initializer):
    def __init__(self, min: Union[int, float], max: Union[int, float]):
        self.min = min
        self.max = max

    def initialize(self, perturbation: torch.Tensor) -> None:
        torch.nn.init.uniform_(perturbation, self.min, self.max)


class UniformLp(Initializer):
    def __init__(self, eps: Union[int, float], p: Optional[Union[int, float]] = torch.inf):
        self.eps = eps
        self.p = p

    def initialize(self, perturbation: torch.Tensor) -> None:
        torch.nn.init.uniform_(perturbation, -self.eps, self.eps)
        # TODO: make sure the first dim is the batch dim.
        if self.p is not torch.inf:
            # We don't do tensor.renorm_() because the first dim is not the batch dim.
            pert_norm = perturbation.data.norm(p=self.p)
            perturbation.data.mul_(self.eps / pert_norm)
