#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch

__all__ = ["Cat", "Permute", "Unsqueeze", "Squeeze", "Chunk", "TupleTransforms"]


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
