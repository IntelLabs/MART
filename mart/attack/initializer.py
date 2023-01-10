#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import abc
from typing import Optional, Union

import torch

__all__ = ["Initializer"]


class Initializer(abc.ABC):
    """Initializer base class."""

    @abc.abstractmethod
    def __call__(self, perturbation: torch.Tensor) -> None:
        pass


class Constant(Initializer):
    def __init__(self, constant: Optional[Union[int, float]] = 0):
        self.constant = constant

    def __call__(self, perturbation: torch.Tensor) -> None:
        torch.nn.init.constant_(perturbation, self.constant)


class Uniform(Initializer):
    def __init__(self, min: Union[int, float], max: Union[int, float]):
        self.min = min
        self.max = max

    def __call__(self, perturbation: torch.Tensor) -> None:
        torch.nn.init.uniform_(perturbation, self.min, self.max)


class UniformLp(Initializer):
    def __init__(self, eps: Union[int, float], p: Optional[Union[int, float]] = torch.inf):
        self.eps = eps
        self.p = p

    def __call__(self, perturbation: torch.Tensor) -> None:
        torch.nn.init.uniform_(perturbation, -self.eps, self.eps)
        # TODO: make sure the first dim is the batch dim.
        if self.p is not torch.inf:
            # We don't do tensor.renorm_() because the first dim is not the batch dim.
            pert_norm = perturbation.data.norm(p=self.p)
            perturbation.data.mul_(self.eps / pert_norm)
