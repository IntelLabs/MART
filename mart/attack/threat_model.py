#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import abc
from typing import Any, Dict, Union

import torch

__all__ = ["BatchThreatModel"]


class ThreatModel(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(
        self,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        perturbation: Union[torch.Tensor, tuple],
        **kwargs
    ) -> Union[torch.Tensor, tuple]:
        raise NotImplementedError


class BatchThreatModel(ThreatModel):
    def __init__(self, threat_model: ThreatModel):
        super().__init__()

        self.threat_model = threat_model

    def forward(
        self,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        perturbation: Union[torch.Tensor, tuple],
        **kwargs
    ) -> Union[torch.Tensor, tuple]:
        output = []

        for input_i, target_i, perturbation_i in zip(input, target, perturbation):
            output_i = self.threat_model(input_i, target_i, perturbation_i, **kwargs)
            output.append(output_i)

        if isinstance(input, torch.Tensor):
            output = torch.stack(output)
        else:
            output = tuple(output)

        return output


class Additive(ThreatModel):
    """We assume an adversary adds perturbation to the input."""

    def forward(
        self,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        perturbation: Union[torch.Tensor, tuple],
        **kwargs
    ) -> Union[torch.Tensor, tuple]:
        return input + perturbation


class Overlay(ThreatModel):
    """We assume an adversary overlays a patch to the input."""

    def forward(
        self,
        input: Union[torch.Tensor, tuple],
        target: Union[torch.Tensor, Dict[str, Any], tuple],
        perturbation: Union[torch.Tensor, tuple],
        **kwargs
    ) -> Union[torch.Tensor, tuple]:
        # True is mutable, False is immutable.
        mask = target["perturbable_mask"]

        # Convert mask to a Tensor with same torch.dtype and torch.device as input,
        #   because some data modules (e.g. Armory) gives binary mask.
        mask = mask.to(input)

        return input * (1 - mask) + perturbation * mask
