#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from typing import Any

import torch
import torchvision
import torchvision.transforms.functional as F

from mart.utils import pylogger

logger = pylogger.get_pylogger(__name__)


class Initializer:
    """Initializer base class."""

    @torch.no_grad()
    def __call__(
        self,
        parameter: torch.Tensor,
        *,
        input: torch.Tensor | None = None,
        target: torch.Tensor | dict[str, Any] | None = None,
    ) -> None:
        # Accept input and target from modality_dispatch().
        self.initialize_(parameter)

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


class Image(Initializer):
    def __init__(self, path: str, scale: int = 1):
        self.image = torchvision.io.read_image(path, torchvision.io.ImageReadMode.RGB) / scale

    @torch.no_grad()
    def initialize_(self, parameter: torch.Tensor) -> None:
        image = self.image

        if image.shape != parameter.shape:
            logger.info(f"Resizing image from {image.shape} to {parameter.shape}...")
            image = F.resize(image, parameter.shape[1:])

        parameter.copy_(image)
