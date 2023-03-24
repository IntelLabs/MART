#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..adversary import Adversary

__all__ = ["Callback"]


class Callback(abc.ABC):
    """Abstract base class of callbacks."""

    def on_run_start(
        self,
        *,
        adversary: Adversary,
        input: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor | dict[str, Any] | list[Any],
        model: torch.nn.Module,
        **kwargs,
    ):
        pass

    def on_examine_start(
        self,
        *,
        adversary: Adversary,
        input: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor | dict[str, Any] | list[Any],
        model: torch.nn.Module,
        **kwargs,
    ):
        pass

    def on_examine_end(
        self,
        *,
        adversary: Adversary,
        input: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor | dict[str, Any] | list[Any],
        model: torch.nn.Module,
        **kwargs,
    ):
        pass

    def on_advance_start(
        self,
        *,
        adversary: Adversary,
        input: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor | dict[str, Any] | list[Any],
        model: torch.nn.Module,
        **kwargs,
    ):
        pass

    def on_advance_end(
        self,
        *,
        adversary: Adversary,
        input: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor | dict[str, Any] | list[Any],
        model: torch.nn.Module,
        **kwargs,
    ):
        pass

    def on_run_end(
        self,
        *,
        adversary: Adversary,
        input: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor | dict[str, Any] | list[Any],
        model: torch.nn.Module,
        **kwargs,
    ):
        pass
