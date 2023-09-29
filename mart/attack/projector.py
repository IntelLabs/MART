#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import abc
from collections import OrderedDict
from typing import Any, Iterable

import torch


class Function:
    def __init__(self, order=0) -> None:
        """A stackable function for Projector.

        Args:
            order (int, optional): The priority number. A smaller number makes a function run earlier than others in a sequence. Defaults to 0.
        """
        self.order = order

    @abc.abstractmethod
    def __call__(self, perturbation, input, target) -> None:
        """It returns None because we only perform non-differentiable in-place operations."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Projector:
    """A projector modifies nn.Parameter's data."""

    def __init__(self, functions: dict[str, Function] = {}) -> None:
        """_summary_

        Args:
            functions (dict[str, Function]): A dictionary of functions for perturbation projection.
        """
        # Sort functions by function.order and the name.
        self.functions_dict = OrderedDict(
            sorted(functions.items(), key=lambda name_fn: (name_fn[1].order, name_fn[0]))
        )
        self.functions = list(self.functions_dict.values())

    def __call__(
        self,
        perturbation: torch.Tensor | Iterable[torch.Tensor],
        *,
        input: torch.Tensor | Iterable[torch.Tensor],
        target: torch.Tensor | Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        **kwargs,
    ) -> None:
        if isinstance(perturbation, torch.Tensor) and isinstance(input, torch.Tensor):
            self.project_(perturbation, input=input, target=target)

        elif (
            isinstance(perturbation, Iterable)
            and isinstance(input, Iterable)  # noqa: W503
            and isinstance(target, Iterable)  # noqa: W503
        ):
            for perturbation_i, input_i, target_i in zip(perturbation, input, target):
                self.project_(perturbation_i, input=input_i, target=target_i)

        else:
            raise NotImplementedError

    @torch.no_grad()
    def project_(
        self,
        perturbation: torch.Tensor | Iterable[torch.Tensor],
        *,
        input: torch.Tensor | Iterable[torch.Tensor],
        target: torch.Tensor | Iterable[torch.Tensor] | Iterable[dict[str, Any]],
    ) -> None:
        for function in self.functions:
            # Some functions such as Mask need access to target["perturbable_mask"]
            function(perturbation, input, target)

    def __repr__(self):
        function_names = [repr(p) for p in self.functions_dict]
        return f"{self.__class__.__name__}({function_names})"


class Range(Function):
    """Clamp the perturbation so that the output is range-constrained.

    Maybe used in overlay composer.
    """

    def __init__(self, quantize: bool = False, min: int | float = 0, max: int | float = 255):
        self.quantize = quantize
        self.min = min
        self.max = max

    def __call__(self, perturbation, input, target):
        if self.quantize:
            perturbation.round_()
        perturbation.clamp_(self.min, self.max)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(quantize={self.quantize}, min={self.min}, max={self.max})"
        )


class Lp(Function):
    """Project perturbations to Lp norm, only if the Lp norm is larger than eps."""

    def __init__(self, eps: int | float, p: int | float = torch.inf, *args, **kwargs):
        """_summary_

        Args:
            eps (float): The max norm.
            p (float): The p in L-p norm, which must be positive.. Defaults to torch.inf.
        """
        super().__init__(*args, **kwargs)

        self.p = p
        self.eps = eps

    @staticmethod
    def linf(x, p, eps):
        x.clamp_(min=-eps, max=eps)

    @staticmethod
    def lp(x, p, eps):
        x_norm = x.norm(p=p)
        if x_norm > eps:
            x.mul_(eps / x_norm)

    def __call__(self, perturbation, input, target):
        if self.p == torch.inf:
            method = self.linf
        elif self.p == 0:
            raise NotImplementedError("L-0 projection is not implemented.")
        else:
            method = self.lp

        method(perturbation, self.p, self.eps)


# TODO: We may move the mask projection to Initialzier, if we also have mask in composer, because no gradient to update the masked pixels.
class Mask(Function):
    def __init__(self, *args, key="perturbable_mask", **kwargs):
        super().__init__(*args, **kwargs)
        self.key = key

    def __call__(self, perturbation, input, target):
        perturbation.mul_(target[self.key])
