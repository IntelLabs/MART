#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from typing import Iterable

import torch

from mart.utils import pylogger

logger = pylogger.get_pylogger(__name__)


class Initializer:
    """Initializer base class."""

    @torch.no_grad()
    def __call__(self, parameters: torch.Tensor | Iterable[torch.Tensor]) -> None:
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        [self.initialize_(parameter) for parameter in parameters]

    @torch.no_grad()
    def initialize_(self, parameter: torch.Tensor) -> None:
        pass


class Constant(Initializer):
    def __init__(self, constant: int | float = 0):
        self.constant = constant

    @torch.no_grad()
    def initialize_(self, parameter: torch.Tensor) -> None:
        torch.nn.init.constant_(parameter, self.constant)


class Uniform(Initializer):
    def __init__(self, min: int | float, max: int | float):
        self.min = min
        self.max = max

    @torch.no_grad()
    def initialize_(self, parameter: torch.Tensor) -> None:
        torch.nn.init.uniform_(parameter, self.min, self.max)


class UniformLp(Initializer):
    def __init__(self, eps: int | float, p: int | float = torch.inf):
        self.eps = eps
        self.p = p

    @torch.no_grad()
    def initialize_(self, parameter: torch.Tensor) -> None:
        torch.nn.init.uniform_(parameter, -self.eps, self.eps)
        # TODO: make sure the first dim is the batch dim.
        if self.p is not torch.inf:
            # We don't do tensor.renorm_() because the first dim is not the batch dim.
            pert_norm = parameter.norm(p=self.p)
            parameter.mul_(self.eps / pert_norm)
