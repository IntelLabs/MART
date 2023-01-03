#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import functools
from typing import Any, Dict, Union

import torch
from hydra.utils import instantiate

from mart.attack.callbacks import Callback

__all__ = ["BatchPerturber"]


class BatchPerturber(Callback, torch.nn.Module):
    """The batch input could be a list or a NCHW tensor.

    We split input into individual examples and run different perturbers accordingly.
    """

    def __init__(self, perturber_factory: functools.partial, *perturber_args, **perturber_kwargs):
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

    def forward(self, input: torch.Tensor, target: Union[torch.Tensor, Dict[str, Any]]) -> None:
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
