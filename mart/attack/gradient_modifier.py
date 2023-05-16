#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from typing import Iterable

import torch

__all__ = ["GradientModifier"]


class GradientModifier:
    """Gradient modifier base class."""

    def __call__(self, parameters: torch.Tensor | Iterable[torch.Tensor]) -> None:
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        [self.modify_(parameter) for parameter in parameters]

    @torch.no_grad()
    def modify_(self, parameter: torch.Tensor) -> None:
        pass


class Sign(GradientModifier):
    @torch.no_grad()
    def modify_(self, parameter: torch.Tensor) -> None:
        parameter.grad.sign_()


class LpNormalizer(GradientModifier):
    """Scale gradients by a certain L-p norm."""

    def __init__(self, p: int | float):
        self.p = float(p)

    @torch.no_grad()
    def modify_(self, parameter: torch.Tensor) -> None:
        p_norm = torch.norm(parameter.grad.detach(), p=self.p)
        parameter.grad.detach().div_(p_norm)
