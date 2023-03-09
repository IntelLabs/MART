#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch
from torchvision.transforms import transforms as T

__all__ = [
    "Denormalize",
    "Cat",
    "Permute",
    "Unsqueeze",
    "Squeeze",
    "Chunk",
    "TupleTransforms",
    "GetItems",
]


class Denormalize(T.Normalize):
    """Unnormalized using center and scale via existing Normalize transform such that:

        output = (input * scale + center)

    Args:
        center: value to center input by
        scale: value to scale input by
    """

    def __init__(self, center, scale, inplace=False):
        mean = -center / scale
        std = 1 / scale

        super().__init__(mean, std, inplace=inplace)


class Cat:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tensors):
        return torch.cat(tensors, dim=self.dim)


class Permute:
    def __init__(self, *dims):
        self.dims = dims

    def __call__(self, tensor):
        return tensor.permute(*self.dims)


class Unsqueeze:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tensor):
        return tensor.unsqueeze(self.dim)


class Squeeze:
    def __init__(self, dim=None):
        self.dim = dim

    def __call__(self, tensor):
        return tensor.squeeze(self.dim)


# TODO: Add Cat transform
# FIXME: Change to Split transform
class Chunk:
    def __init__(self, chunks, dim=0):
        self.chunks = chunks
        self.dim = dim

    def __call__(self, tensor):
        chunks = tensor.chunk(self.chunks, dim=self.dim)
        # print("chunks =", type(chunks))
        return [*chunks]  # tuple -> list


class TupleTransforms(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()

        self.transforms = transforms

    def forward(self, x_tuple):
        output_tuple = tuple(self.transforms(x) for x in x_tuple)
        return output_tuple


class GetItems:
    """Get a list of values with a list of keys from a dictionary."""

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, x):
        x_list = [x[key] for key in self.keys]
        return x_list
