#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import abc

import torch

__all__ = ["Objective"]


class Objective(abc.ABC):
    """Objectives do not need to be differentiable so we do not inherit from nn.Module."""

    @abc.abstractmethod
    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
