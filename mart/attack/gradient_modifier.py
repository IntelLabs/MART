#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import abc

__all__ = ["GradientModifier"]


class GradientModifier(abc.ABC):
    """Gradient modifier base class."""

    @abc.abstractmethod
    def __call__(self, grad):
        pass


class Sign(GradientModifier):
    def __call__(self, grad):
        return grad.sign()


class LpNormalizer(GradientModifier):
    """Scale gradients by a certain L-p norm."""

    def __init__(self, p):
        super().__init__

        self.p = p

    def __call__(self, grad):
        grad_norm = grad.norm(p=self.p)
        grad_normalized = grad / grad_norm
        return grad_normalized
