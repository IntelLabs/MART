#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import typing
from typing import Any, Callable

import torch

from ..callbacks import Callback

if typing.TYPE_CHECKING:
    from ..adversary import Adversary
    from ..gradient_modifier import GradientModifier
    from ..initializer import Initializer
    from ..projector import Projector
    from .perturber import Perturber

__all__ = ["BatchPerturber"]


class BatchPerturber(Callback, torch.nn.Module):
    """The batch input could be a list or a NCHW tensor.

    We split input into individual examples and run different perturbers accordingly.
    """

    def __init__(
        self,
        perturber_factory: Callable[[Initializer, GradientModifier, Projector], Perturber],
        *perturber_args,
        **perturber_kwargs,
    ):
        super().__init__()

        self.perturber_factory = perturber_factory
        self.perturber_args = perturber_args
        self.perturber_kwargs = perturber_kwargs

        # Try to create a perturber using factory and kwargs
        assert self.perturber_factory(*self.perturber_args, **self.perturber_kwargs) is not None

        self.perturbers = torch.nn.ModuleDict()

    def on_run_start(self, adversary, input, target, model, **kwargs):
        # Remove old perturbers
        # FIXME: Can we do this in on_run_end instead?
        self.perturbers.clear()

        # Create new perturber for each item in the batch
        for i in range(len(input)):
            perturber = self.perturber_factory(*self.perturber_args, **self.perturber_kwargs)
            self.perturbers[f"input_{i}_perturber"] = perturber

        # Trigger callback
        for i, (input_i, target_i) in enumerate(zip(input, target)):
            perturber = self.perturbers[f"input_{i}_perturber"]
            if isinstance(perturber, Callback):
                perturber.on_run_start(
                    adversary=adversary, input=input_i, target=target_i, model=model, **kwargs
                )

    def on_advance_start(
        self,
        *,
        adversary: Adversary,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        model: torch.nn.Module,
        **kwargs,
    ):
        for i, (input_i, target_i) in enumerate(zip(input, target)):
            perturber = self.perturbers[f"input_{i}_perturber"]
            if isinstance(perturber, Callback):
                perturber.on_advance_start(
                    adversary=adversary, input=input_i, target=target_i, model=model, **kwargs
                )

    def on_advance_end(
        self,
        *,
        adversary: Adversary,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        model: torch.nn.Module,
        **kwargs,
    ):
        for i, (input_i, target_i) in enumerate(zip(input, target)):
            perturber = self.perturbers[f"input_{i}_perturber"]
            if isinstance(perturber, Callback):
                perturber.on_advance_end(
                    adversary=adversary, input=input_i, target=target_i, model=model, **kwargs
                )

    def forward(self, input: torch.Tensor, target: torch.Tensor | dict[str, Any]) -> None:
        output = []
        for i, (input_i, target_i) in enumerate(zip(input, target)):
            perturber = self.perturbers[f"input_{i}_perturber"]
            ret_i = perturber(input_i, target_i)
            output.append(ret_i)

        if isinstance(input, torch.Tensor):
            output = torch.stack(output)
        else:
            output = tuple(output)

        return output
