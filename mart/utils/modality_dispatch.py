#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import functools
from itertools import cycle
from typing import Any, Callable, Iterable

import torch

__all__ = ["modality_dispatch"]

DEFAULT_MODALITY = "default"


# We make input the first non-keyword argument for singledispatch to work.
@functools.singledispatch
def modality_dispatch(
    input: torch.Tensor | dict[str, torch.Tensor] | Iterable[Any],
    *,
    data: torch.Tensor | dict[str, torch.Tensor] | Iterable[Any],
    target: torch.Tensor | Iterable[Any] | None,
    modality_func: Callable | dict[str, Callable],
    modality: str = DEFAULT_MODALITY,
):
    """Recursively dispatch data and input/target to functions of the same modality.

    The function returns an object that is homomorphic to input.
    """

    raise ValueError(
        f"Unsupported data type combination: type(input)={type(input)} and type(data)={type(data)}."
    )

    # if (
    #     isinstance(input, torch.Tensor)
    #     and isinstance(data, torch.Tensor)  # noqa: W503
    #     and isinstance(modality_func, dict)  # noqa: W503
    # ):
    #     # A dictionary of Callable indexed by modality.
    #     return modality_func[modality](data, input=input, target=target)
    # elif (
    #     isinstance(input, torch.Tensor)
    #     and isinstance(data, torch.Tensor)  # noqa: W503
    #     and isinstance(modality_func, Callable)  # noqa: W503
    # ):
    #     # A Callable with modality=? as a keyword argument.
    #     return modality_func(data, input=input, target=target, modality=modality)
    # elif isinstance(input, dict):
    #     # The dict input has modalities specified in keys, passing them recursively.
    #     output = {}
    #     for modality in input.keys():
    #         output[modality] = modality_dispatch(
    #             modality_func,
    #             data[modality],
    #             input=input[modality],
    #             target=target,
    #             modality=modality,
    #         )
    #     return output
    # elif isinstance(input, (list, tuple)):
    #     # The list or tuple input is a collection of sub-input and sub-target.
    #     if not isinstance(target, (list, tuple)):
    #         # Make target zip well with input.
    #         target = cycle([target])
    #     if not isinstance(data, (list, tuple)):
    #         # Data is shared for all input, e.g. universal perturbation.
    #         # Make data zip well with input.
    #         data = cycle([data])

    #     output = []
    #     for data_i, input_i, target_i in zip(data, input, target):
    #         output_i = modality_dispatch(
    #             modality_func, data_i, input=input_i, target=target_i, modality=modality
    #         )
    #         output.append(output_i)
    #     if isinstance(input, tuple):
    #         output = tuple(output)
    #     return output
    # else:
    #     raise ValueError(
    #         f"Unsupported data type combination: type(input)={type(input)} and type(data)={type(data)}."
    #     )


@modality_dispatch.register
def _(input: torch.Tensor, *, data, target, modality, modality_func):
    if isinstance(modality_func, dict):
        # A dictionary of Callable indexed by modality.
        return modality_func[modality](data, input=input, target=target)
    elif isinstance(modality_func, Callable):
        # A Callable with modality=? as a keyword argument.
        return modality_func(data, input=input, target=target, modality=modality)


@modality_dispatch.register
def _(input: dict, *, data, target, modality, modality_func):
    # The dict input has modalities specified in keys, passing them recursively.
    output = {}
    for modality in input.keys():
        output[modality] = modality_dispatch(
            input[modality],
            data=data[modality],
            target=target,
            modality=modality,
            modality_func=modality_func,
        )
    return output


@modality_dispatch.register
def _(input: list, *, data, target, modality, modality_func):
    # The list input implies a collection of sub-input and sub-target.
    if not isinstance(target, (list, tuple)):
        # Make target zip well with input.
        target = cycle([target])
    if not isinstance(data, (list, tuple)):
        # Make data zip well with input.
        # Data is shared for all input, e.g. universal perturbation.
        data = cycle([data])

    output = []
    for data_i, input_i, target_i in zip(data, input, target):
        output_i = modality_dispatch(
            input_i,
            data=data_i,
            target=target_i,
            modality=modality,
            modality_func=modality_func,
        )
        output.append(output_i)

    return output


@modality_dispatch.register
def _(input: tuple, *, data, target, modality, modality_func):
    # The tuple input is similar with the list input.
    output = modality_dispatch(
        list(input),
        data=data,
        target=target,
        modality=modality,
        modality_func=modality_func,
    )
    # Make the output a tuple, the same as input.
    output = tuple(output)
    return output
