#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

__all__ = ["Denormalize"]

from torchvision.transforms import transforms as T


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
