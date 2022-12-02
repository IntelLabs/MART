#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import abc

import torch

from .perturber import Perturber

__all__ = ["Projector"]


class Projector(abc.ABC):
    """A projector modifies nn.Parameter's data."""

    @abc.abstractmethod
    def __call__(self, tensor, input, target):
        pass


class Compose(Projector):
    """Apply a list of perturbation modifier."""

    def __init__(self, projectors):
        self.projectors = projectors

    def __call__(self, tensor, input, target):
        for projector in self.projectors:
            projector(tensor, input, target)

    def __repr__(self):
        projector_names = [repr(p) for p in self.projectors]
        return f"{self.__class__.__name__}({projector_names})"


class Range(Projector):
    """Clamp the perturbation so that the output is range-constrained."""

    def __init__(self, quantize=False, min=0, max=255):
        self.quantize = quantize
        self.min = min
        self.max = max

    def __call__(self, tensor, input, target):
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

    def __init__(self, quantize=False, min=0, max=255):
        self.quantize = quantize
        self.min = min
        self.max = max

    def __call__(self, tensor, input, target):
        if self.quantize:
            tensor.round_()
        tensor.clamp_(self.min - input, self.max - input)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(quantize={self.quantize}, min={self.min}, max={self.max})"
        )


class Lp(Projector):
    """Project perturbations to Lp norm, only if the Lp norm is larger than eps."""

    def __init__(self, eps, p=torch.inf):
        """_summary_

        Args:
            eps (float): The max norm.
            p (float): The p in L-p norm, which must be positive.. Defaults to torch.inf.
        """

        self.p = p
        self.eps = eps

    def __call__(self, tensor, input, target):
        pert_norm = tensor.norm(p=self.p)
        if pert_norm > self.eps:
            # We only upper-bound the norm.
            tensor.mul_(self.eps / pert_norm)


class LinfAdditiveRange(Projector):
    """Make sure the perturbation is within the Linf norm ball, and "input + perturbation" is
    within the [min, max] range."""

    def __init__(self, eps, min=0, max=255):
        self.eps = eps
        self.min = min
        self.max = max

    def __call__(self, tensor, input, target):
        eps_min = (input - self.eps).clamp(self.min, self.max) - input
        eps_max = (input + self.eps).clamp(self.min, self.max) - input

        tensor.clamp_(eps_min, eps_max)


class Mask(Projector):
    def __call__(self, tensor, input, target):
        tensor.mul_(target["perturbable_mask"])

    def __repr__(self):
        return f"{self.__class__.__name__}()"
