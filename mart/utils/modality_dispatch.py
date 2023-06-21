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

DEFAULT_MODALITY = "default"


@functools.singledispatch
def modality_dispatch(
    input: torch.Tensor | Iterable[torch.Tensor] | Iterable[dict[str, torch.Tensor]],
    *,
    data: torch.Tensor | Iterable[torch.Tensor] | Iterable[dict[str, torch.Tensor]],
    target: torch.Tensor | Iterable[torch.Tensor] | Iterable[dict[str, torch.Tensor | str]],
    modality_func: Callable | dict[str, Callable],
    modality: str = DEFAULT_MODALITY,
):
    """Recursively dispatch data and input/target to functions of the same modality.

    The function returns an object that is homomorphic to input. We make input the first non-
    keyword argument for singledispatch to work.
    """

    raise ValueError(f"Unsupported data type of input: type(input)={type(input)}.")


@modality_dispatch.register
def _(input: torch.Tensor, *, data, target, modality_func, modality=DEFAULT_MODALITY):
    # Take action when input is a tensor.
    if isinstance(modality_func, dict):
        # A dictionary of Callable indexed by modality.
        return modality_func[modality](data, input=input, target=target)
    elif isinstance(modality_func, Callable):
        # A Callable with modality=? as a keyword argument.
        return modality_func(data, input=input, target=target, modality=modality)


@modality_dispatch.register
def _(input: dict, *, data, target, modality_func, modality=DEFAULT_MODALITY):
    # The dict input has modalities specified in keys, passing them recursively.
    output = {}
    for modality in input.keys():
        output[modality] = modality_dispatch(
            input[modality],
            data=data[modality],
            target=target,
            modality_func=modality_func,
            modality=modality,
        )
    return output


@modality_dispatch.register
def _(input: list, *, data, target, modality_func, modality=DEFAULT_MODALITY):
    # The list input implies a collection of sub-input and sub-target.
    if not isinstance(target, Iterable):
        # Make target zip well with input.
        target = cycle([target])
    if not isinstance(data, Iterable):
        # Make data zip well with input.
        # Besides list and tuple, data could be ParameterList too.
        # Data is shared for all input, e.g. universal perturbation.
        data = cycle([data])

    output = []
    for data_i, input_i, target_i in zip(data, input, target):
        output_i = modality_dispatch(
            input_i,
            data=data_i,
            target=target_i,
            modality_func=modality_func,
            modality=modality,
        )
        output.append(output_i)

    return output


@modality_dispatch.register
def _(input: tuple, *, data, target, modality_func, modality=DEFAULT_MODALITY):
    # The tuple input is similar with the list input.
    output = modality_dispatch(
        list(input),
        data=data,
        target=target,
        modality_func=modality_func,
        modality=modality,
    )
    # Make the output a tuple, the same as input.
    output = tuple(output)
    return output


class ModalityParameterDict(torch.nn.ParameterDict):
    """Get a new name so we know when parameters are associated with modality."""

    pass
