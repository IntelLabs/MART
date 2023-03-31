#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor

__all__ = ["modality_dispatch"]


def modality_dispatch(
    modality_func: dict[str, Callable],
    data: Tensor | tuple | list[Tensor] | dict[str, Tensor],
    *,
    input: Tensor | tuple | list[Tensor] | dict[str, Tensor],
    target: torch.Tensor | dict[str, Any],
    modality: str = "default",
):
    """Recursively dispatch data and input/target to functions of the same modality.

    The function returns an object that is homomorphic to input and data.
    """

    assert type(data) == type(input)

    if isinstance(input, torch.Tensor):
        return modality_func[modality](data, input=input, target=target)
    elif isinstance(input, dict):
        # The dict input has modalities specified in keys, passing them recursively.
        output = {}
        for modality in input.keys():
            output[modality] = modality_dispatch(
                modality_func,
                data[modality],
                input=input[modality],
                target=target,
                modality=modality,
            )
        return output
    elif isinstance(input, (list, tuple)):
        # The list or tuple input is a collection of sub-input and sub-target.
        output = []
        for data_i, input_i, target_i in zip(data, input, target):
            output_i = modality_dispatch(
                modality_func, data_i, input=input_i, target=target_i, modality=modality
            )
            output.append(output_i)
        if isinstance(input, tuple):
            output = tuple(output)
        return output
    else:
        raise ValueError(f"Unsupported data type of input: {type(input)}.")
