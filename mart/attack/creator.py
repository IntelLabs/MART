#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import abc

import torch

__all__ = ["InputDuplicator", "PatchCreator"]


class Creator(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *, input, target):
        raise NotImplementedError


class InputDuplicator(Creator):
    def __call__(self, *, input, target):
        # TODO: we don't want an integer tensor, but make sure it does not affect mixed precision training.
        pert = torch.empty_like(input, dtype=torch.float, requires_grad=True)
        return pert


class PatchCreator(Creator):
    def __call__(self, *, input, target):
        coords = target["patch_coords"]
        leading_dims = list(input.shape[:-2])
        width = coords[:, 0].max() - coords[:, 0].min()
        height = coords[:, 1].max() - coords[:, 1].min()
        shape = list(leading_dims) + [height, width]
        pert = torch.empty(shape, device=input.device, dtype=torch.float, requires_grad=True)
        return pert
