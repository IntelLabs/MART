#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from typing import Any, Callable

import torch

__all__ = ["NormalizedAdversaryAdapter"]


class NormalizedAdversaryAdapter(torch.nn.Module):
    """A wrapper for running external classification adversaries in MART.

    External adversaries commonly take input of NCWH-[0,1] and return input_adv in the same format.
    """

    def __init__(
        self,
        adversary: Callable[[Callable], Callable],
        enforcer: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
    ):
        """

        Args:
            adversary (functools.partial): A partial of an adversary object which awaits model.
            enforcer (Callable): Enforcing constraints of an adversary.
        """
        super().__init__()

        self.adversary = adversary
        self.enforcer = enforcer

    def forward(
        self,
        input: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor | dict[str, Any] | list[Any],
        model: torch.nn.Module | None = None,
        **kwargs,
    ):

        # Shortcut. Input is already updated in the attack loop.
        if model is None:
            return input

        # Input NCHW [0,1]; Output logits.
        def model_wrapper(x):
            output = model(input=x * 255, target=target, model=None, **kwargs)
            logits = output["logits"]
            return logits

        attack = self.adversary(model_wrapper)
        input_adv = attack(input / 255, target)

        # Round to integer, in case of imprecise scaling.
        input_adv = (input_adv * 255).round()
        self.enforcer(input_adv, input=input, target=target)

        return input_adv
