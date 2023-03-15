#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import abc
from typing import Any, Dict, List, Optional, Union

import torch

__all__ = ["Projector"]


class Projector(abc.ABC):
    """A projector modifies nn.Parameter's data."""

    def __call__(
        self,
        tensor: torch.Tensor,
        input: torch.Tensor,
        target: Union[torch.Tensor, Dict[str, Any]],
    ) -> None:
        pass


class Compose(Projector):
    """Apply a list of perturbation modifier."""

    def __init__(self, projectors: List[Projector]):
        self.projectors = projectors

    def __call__(
        self,
        tensor: torch.Tensor,
        input: torch.Tensor,
        target: Union[torch.Tensor, Dict[str, Any]],
    ) -> None:
        for projector in self.projectors:
            projector(tensor, input, target)

    def __repr__(self):
        projector_names = [repr(p) for p in self.projectors]
        return f"{self.__class__.__name__}({projector_names})"


class Range(Projector):
    """Clamp the perturbation so that the output is range-constrained."""

    def __init__(
        self,
        quantize: Optional[bool] = False,
        min: Optional[Union[int, float]] = 0,
        max: Optional[Union[int, float]] = 255,
    ):
        self.quantize = quantize
        self.min = min
        self.max = max

    def __call__(
        self,
        tensor: torch.Tensor,
        input: torch.Tensor,
        target: Union[torch.Tensor, Dict[str, Any]],
    ) -> None:
        if self.quantize:
            tensor.round_()
        tensor.clamp_(self.min, self.max)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(quantize={self.quantize}, min={self.min}, max={self.max})"
        )


class RangeAdditive(Projector):
    """Clamp the perturbation so that the output is range-constrained.

    The projector assumes an additive perturbation threat model.
    """

    def __init__(
        self,
        quantize: Optional[bool] = False,
        min: Optional[Union[int, float]] = 0,
        max: Optional[Union[int, float]] = 255,
    ):
        self.quantize = quantize
        self.min = min
        self.max = max

    def __call__(
        self,
        tensor: torch.Tensor,
        input: torch.Tensor,
        target: Union[torch.Tensor, Dict[str, Any]],
    ) -> None:
        if self.quantize:
            tensor.round_()
        tensor.clamp_(self.min - input, self.max - input)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(quantize={self.quantize}, min={self.min}, max={self.max})"
        )


class Lp(Projector):
    """Project perturbations to Lp norm, only if the Lp norm is larger than eps."""

    def __init__(self, eps: float, p: Optional[Union[int, float]] = torch.inf):
        """_summary_

        Args:
            eps (float): The max norm.
            p (float): The p in L-p norm, which must be positive.. Defaults to torch.inf.
        """

        self.p = p
        self.eps = eps

    def __call__(
        self,
        tensor: torch.Tensor,
        input: torch.Tensor,
        target: Union[torch.Tensor, Dict[str, Any]],
    ) -> None:
        pert_norm = tensor.norm(p=self.p)
        if pert_norm > self.eps:
            # We only upper-bound the norm.
            tensor.mul_(self.eps / pert_norm)


class LinfAdditiveRange(Projector):
    """Make sure the perturbation is within the Linf norm ball, and "input + perturbation" is
    within the [min, max] range."""

    def __init__(
        self,
        eps: float,
        min: Optional[Union[int, float]] = 0,
        max: Optional[Union[int, float]] = 255,
    ):
        self.eps = eps
        self.min = min
        self.max = max

    def __call__(
        self,
        tensor: torch.Tensor,
        input: torch.Tensor,
        target: Union[torch.Tensor, Dict[str, Any]],
    ) -> None:
        eps_min = (input - self.eps).clamp(self.min, self.max) - input
        eps_max = (input + self.eps).clamp(self.min, self.max) - input

        tensor.clamp_(eps_min, eps_max)


class Mask(Projector):
    def __call__(
        self,
        tensor: torch.Tensor,
        input: torch.Tensor,
        target: Union[torch.Tensor, Dict[str, Any]],
    ) -> None:
        tensor.mul_(target["perturbable_mask"])

    def __repr__(self):
        return f"{self.__class__.__name__}()"
