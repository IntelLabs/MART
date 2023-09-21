#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from functools import singledispatch

import numpy as np
import torch


# A recursive function to convert all np.ndarray in an object to torch.Tensor, or vice versa.
@singledispatch
def convert(obj, device=None):
    """All other types, no change."""
    return obj


@convert.register
def _(obj: dict, device=None):
    return {key: convert(value, device=device) for key, value in obj.items()}


@convert.register
def _(obj: list, device=None):
    return [convert(item, device=device) for item in obj]


@convert.register
def _(obj: tuple, device=None):
    return tuple(convert(obj, device=device))


@convert.register
def _(obj: np.ndarray, device=None):
    return torch.tensor(obj, device=device)


@convert.register
def _(obj: torch.Tensor, device=None):
    return obj.detach().cpu().numpy()
