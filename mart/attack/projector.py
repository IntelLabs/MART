#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from typing import Any

import torch


class Projector:
    """A projector modifies nn.Parameter's data."""

    @torch.no_grad()
    def __call__(
        self,
        perturbation: torch.Tensor | tuple,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        **kwargs,
    ) -> None:
        if isinstance(perturbation, tuple) and isinstance(input, tuple):
            for perturbation_i, input_i, target_i in zip(perturbation, input, target):
                self.project(perturbation_i, input=input_i, target=target_i)

        elif isinstance(perturbation, torch.Tensor) and isinstance(input, tuple):
            for input_i, target_i in zip(input, target):
                self.project(perturbation, input=input_i, target=target_i)

        elif isinstance(perturbation, torch.Tensor) and isinstance(input, torch.Tensor):
            self.project(perturbation, input=input, target=target)

        else:
            raise NotImplementedError

    def project(
        self,
        perturbation: torch.Tensor | tuple,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
    ) -> None:
        pass


class Compose(Projector):
    """Apply a list of perturbation modifier."""

    def __init__(self, projectors: list[Projector]):
        self.projectors = projectors

    @torch.no_grad()
    def __call__(
        self,
        perturbation: torch.Tensor | tuple,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        **kwargs,
    ) -> None:
        for projector in self.projectors:
            projector(perturbation, input=input, target=target)

    def __repr__(self):
        projector_names = [repr(p) for p in self.projectors]
        return f"{self.__class__.__name__}({projector_names})"


class Range(Projector):
    """Clamp the perturbation so that the output is range-constrained."""

    def __init__(self, quantize: bool = False, min: int | float = 0, max: int | float = 255):
        self.quantize = quantize
        self.min = min
        self.max = max

    def project(self, perturbation, *, input, target):
        if self.quantize:
            perturbation.round_()
        perturbation.clamp_(self.min, self.max)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(quantize={self.quantize}, min={self.min}, max={self.max})"
        )


class RangeAdditive(Projector):
    """Clamp the perturbation so that the output is range-constrained.

    The projector assumes an additive perturbation threat model.
    """

    def __init__(self, quantize: bool = False, min: int | float = 0, max: int | float = 255):
        self.quantize = quantize
        self.min = min
        self.max = max

    def project(self, perturbation, *, input, target):
        if self.quantize:
            perturbation.round_()
        perturbation.clamp_(self.min - input, self.max - input)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(quantize={self.quantize}, min={self.min}, max={self.max})"
        )


class Lp(Projector):
    """Project perturbations to Lp norm, only if the Lp norm is larger than eps."""

    def __init__(self, eps: int | float, p: int | float = torch.inf):
        """_summary_

        Args:
            eps (float): The max norm.
            p (float): The p in L-p norm, which must be positive.. Defaults to torch.inf.
        """

        self.p = p
        self.eps = eps

    def project(self, perturbation, *, input, target):
        pert_norm = perturbation.norm(p=self.p)
        if pert_norm > self.eps:
            # We only upper-bound the norm.
            perturbation.mul_(self.eps / pert_norm)


class LinfAdditiveRange(Projector):
    """Make sure the perturbation is within the Linf norm ball, and "input + perturbation" is
    within the [min, max] range."""

    def __init__(self, eps: int | float, min: int | float = 0, max: int | float = 255):
        self.eps = eps
        self.min = min
        self.max = max

    def project(self, perturbation, *, input, target):
        eps_min = (input - self.eps).clamp(self.min, self.max) - input
        eps_max = (input + self.eps).clamp(self.min, self.max) - input

        perturbation.clamp_(eps_min, eps_max)


class Mask(Projector):
    def project(self, perturbation, *, input, target):
        perturbation.mul_(target["perturbable_mask"])

    def __repr__(self):
        return f"{self.__class__.__name__}()"
