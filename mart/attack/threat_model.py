#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import abc

import torch

__all__ = ["BatchThreatModel"]


class ThreatModel(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, input, target, perturbation, **kwargs):
        raise NotImplementedError


class BatchThreatModel(ThreatModel):
    def __init__(self, threat_model):
        super().__init__()

        self.threat_model = threat_model

    def forward(self, input, target, perturbation, **kwargs):
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

    def forward(self, input, target, perturbation, **kwargs):
        return input + perturbation


class Overlay(ThreatModel):
    """We assume an adversary overlays a patch to the input."""

    def forward(self, input, target, perturbation, **kwargs):
        # True is mutable, False is immutable.
        mask = target["perturbable_mask"]

        # Convert mask to a Tensor with same torch.dtype and torch.device as input,
        #   because some data modules (e.g. Armory) gives binary mask.
        mask = mask.to(input)

        return input * (1 - mask) + perturbation
