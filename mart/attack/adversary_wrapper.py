#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import torch

__all__ = ["NormalizedAdversaryAdapter"]


class NormalizedAdversaryAdapter(torch.nn.Module):
    """A wrapper for running external classification adversaries in MART.

    External adversaries commonly take input of NCWH-[0,1] and return input_adv in the same format.
    """

    def __init__(self, adversary):
        """

        Args:
            adversary (functools.partial): A partial of an adversary object which awaits model.
        """
        super().__init__()

        self.adversary = adversary

    def forward(self, input, target, model=None, **kwargs):

        # Shortcut. Input is already updated in the attack loop.
        if model is None:
            return input

        # Input NCHW [0,1]; Output logits.
        def model_wrapper(x):
            output = model(input=x * 255, target=target, model=None, **kwargs)
            logits = output["logits"]
            return logits

        attack = self.adversary(model_wrapper)

        return attack(input / 255, target) * 255
