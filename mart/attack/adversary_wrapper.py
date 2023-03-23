#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from typing import Any, Callable

import torch

__all__ = ["NormalizedAttackerAdapter"]


class NormalizedAttackerAdapter(torch.nn.Module):
    """A wrapper for running external classification adversaries in MART.

    External attack algorithms commonly take input of NCWH-[0,1] and return input_adv in the same
    format.
    """

    def __init__(
        self,
        attacker: Callable[[Callable], Callable],
    ):
        """

        Args:
            attacker (functools.partial): A partial of an attacker object which awaits a model.
        """
        super().__init__()

        self.attacker = attacker
        self.input_adv = None

    def forward(
        self,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
    ):
        # Return adversarial input if it is already updated in the attack loop.
        if self.input_adv is None:
            return input
        else:
            return self.input_adv

    def fit(self, *, input, target, model, **kwargs):
        # Input NCHW [0,1]; Output logits.
        def model_wrapper(x):
            output = model(input=x * 255, target=target, model=None, **kwargs)
            logits = output["logits"]
            return logits

        attack = self.attacker(model_wrapper)
        input_adv = attack(input / 255, target)

        # Round to integer, in case of imprecise scaling.
        input_adv = (input_adv * 255).round()

        # Save to return later in forward().
        self.input_adv = input_adv

        return input_adv
