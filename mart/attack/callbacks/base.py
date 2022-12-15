#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import abc
from typing import Any, Dict, Union

import torch

from mart.nn import SequentialDict

__all__ = ["Callback"]


class Callback(abc.ABC):
    """Abstract base class of callbacks."""

    def on_run_start(
        self,
        adversary: torch.nn.Module,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        model: SequentialDict,
        **kwargs
    ):
        pass

    def on_examine_start(
        self,
        adversary: torch.nn.Module,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        model: SequentialDict,
        **kwargs
    ):
        pass

    def on_examine_end(
        self,
        adversary: torch.nn.Module,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        model: SequentialDict,
        **kwargs
    ):
        pass

    def on_advance_start(
        self,
        adversary: torch.nn.Module,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        model: SequentialDict,
        **kwargs
    ):
        pass

    def on_advance_end(
        self,
        adversary: torch.nn.Module,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        model: SequentialDict,
        **kwargs
    ):
        pass

    def on_run_end(
        self,
        adversary: torch.nn.Module,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        model: SequentialDict,
        **kwargs
    ):
        pass
