#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import hydra
import numpy as np
import torch
from multimethod import multimethod
from omegaconf import OmegaConf


# A recursive function to convert all np.ndarray in an object to torch.Tensor, or vice versa.
@multimethod
def convert(obj: dict, device=None):
    return {key: convert(value) for key, value in obj.items()}


@multimethod
def convert(obj: list, device=None):  # noqa: F811
    return [convert(item) for item in obj]


@multimethod
def convert(obj: tuple, device=None):  # noqa: F811
    return tuple(convert(obj))


@multimethod
def convert(obj: np.ndarray, device=None):  # noqa: F811
    return torch.tensor(obj, device=device)


@multimethod
def convert(obj: torch.Tensor, device=None):  # noqa: F811
    return obj.detach().cpu().numpy()


# All other types, no change.
@multimethod
def convert(obj, device=None):  # noqa: F811
    return obj


class MartAttack:
    """A minimal wrapper to run PyTorch-based MART adversary in Armory against PyTorch-based
    models.

    1. Extract the PyTorch model from an ART Estimator;
    2. Convert np.ndarray to torch.Tensor;
    3. Run PyTorch-based MART adversary and get result as torch.Tensor;
    4. Convert torch.Tensor back to np.ndarray.
    """

    def __init__(self, model, mart_adv_config_yaml):
        # TODO: Automatically search for torch.nn.Module within model.
        # Extract PyTorch model from an ART Estimator.
        self.model = model._model
        self.device = self.model.device

        # Instantiate a MART adversary.
        adv_cfg = OmegaConf.load(mart_adv_config_yaml)
        self.adversary = hydra.utils.instantiate(adv_cfg)

    def generate(self, **batch_np):
        # Convert np.ndarray to torch.Tensor.
        # Specify a device to place PyTorch tensors.
        batch_pth = convert(batch_np, device=self.device)
        batch_adv_pytorch = self.adversary(batch_pth, model=self.model)

        # Convert torch.Tensor to np.ndarray.
        batch_adv_np = convert(batch_adv_pytorch)

        # Only return adversarial input in the original numpy format.
        input_adv_np = batch_adv_np["x"]
        return input_adv_np
