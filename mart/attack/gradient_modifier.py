#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import abc
from typing import Union

import torch

__all__ = ["GradientModifier"]


class GradientModifier(abc.ABC):
    """Gradient modifier base class."""

    @abc.abstractmethod
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        pass


class Sign(GradientModifier):
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        return grad.sign()


class LpNormalizer(GradientModifier):
    """Scale gradients by a certain L-p norm."""

    def __init__(self, p: Union[int, float]):
        super().__init__

        self.p = p

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        grad_norm = grad.norm(p=self.p)
        grad_normalized = grad / grad_norm
        return grad_normalized
