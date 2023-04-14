#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import abc
from typing import Iterable

import torch

__all__ = ["GradientModifier"]


class GradientModifier(abc.ABC):
    """Gradient modifier base class."""

    def __call__(self, parameters: torch.Tensor | Iterable[torch.Tensor]) -> None:
        pass


class Sign(GradientModifier):
    def __call__(self, parameters: torch.Tensor | Iterable[torch.Tensor]) -> None:
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        parameters = [p for p in parameters if p.grad is not None]

        for p in parameters:
            p.grad.detach().sign_()


class LpNormalizer(GradientModifier):
    """Scale gradients by a certain L-p norm."""

    def __init__(self, p: int | float):
        self.p = float(p)

    def __call__(self, parameters: torch.Tensor | Iterable[torch.Tensor]) -> None:
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        parameters = [p for p in parameters if p.grad is not None]

        for p in parameters:
            p_norm = torch.norm(p.grad.detach(), p=self.p)
            p.grad.detach().div_(p_norm)
