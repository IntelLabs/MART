#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from itertools import cycle
from typing import Any, Callable, Iterable

import torch

__all__ = ["modality_dispatch"]

DEFAULT_MODALITY = "default"


def modality_dispatch(
    modality_func: Callable | dict[str, Callable],
    data: torch.Tensor | dict[str, torch.Tensor] | Iterable[Any],
    *,
    input: torch.Tensor | dict[str, torch.Tensor] | Iterable[Any],
    target: torch.Tensor | Iterable[Any] | None,
    modality: str = DEFAULT_MODALITY,
):
    """Recursively dispatch data and input/target to functions of the same modality.

    The function returns an object that is homomorphic to input and data.
    """

    if target is None:
        # Make target zips well with input.
        target = cycle([None])

    if (
        isinstance(input, torch.Tensor)
        and isinstance(data, torch.Tensor)  # noqa: W503
        and isinstance(modality_func, dict)  # noqa: W503
    ):
        # A dictionary of Callable indexed by modality.
        return modality_func[modality](data, input=input, target=target)
    elif (
        isinstance(input, torch.Tensor)
        and isinstance(data, torch.Tensor)  # noqa: W503
        and isinstance(modality_func, Callable)  # noqa: W503
    ):
        # A Callable with modality=? as a keyword argument.
        return modality_func(data, input=input, target=target, modality=modality)
    elif isinstance(input, dict) and isinstance(data, dict):
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
    elif isinstance(input, (list, tuple)) and isinstance(data, (list, tuple)):
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
    elif isinstance(input, (list, tuple)) and isinstance(data, torch.Tensor):
        # Data is shared for all input, e.g. universal perturbation.
        output = []
        for input_i, target_i in zip(input, target):
            output_i = modality_dispatch(
                modality_func, data, input=input_i, target=target_i, modality=modality
            )
            output.append(output_i)
        if isinstance(input, tuple):
            output = tuple(output)
        return output
    else:
        raise ValueError(
            f"Unsupported data type combination: type(input)={type(input)} and type(data)={type(data)}."
        )
