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

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        pass


# FIXME: We should really take inspiration from torch.nn.utils.clip_grad_norm_
class Sign(GradientModifier):
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        return grad.sign()


# FIXME: We should really take inspiration from torch.nn.utils.clip_grad_norm_
class LpNormalizer(GradientModifier):
    """Scale gradients by a certain L-p norm."""

    def __init__(self, p: Union[int, float]):
        super().__init__

        self.p = p

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        grad_norm = grad.norm(p=self.p)
        grad_normalized = grad / grad_norm
        return grad_normalized
