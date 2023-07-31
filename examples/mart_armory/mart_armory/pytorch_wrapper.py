#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import hydra
import torch
from omegaconf import OmegaConf

from mart.models.dual_mode import DualModeGeneralizedRCNN


class ArtRcnnModelWrapper(torch.nn.Module):
    """Modify the model so that it is convenient to attack.

    Common issues:
        1. Make the model accept a single argument `output=model(batch)`;
        2. Make the model return loss in eval mode;
        3. Change non-differentiable operations.
    """

    def __init__(self, model):
        super().__init__()

        # Extract PyTorch model from an ART Estimator.
        # TODO: Automatically search for torch.nn.Module within model.
        self.model = DualModeGeneralizedRCNN(model._model)

    def forward(self, batch):
        # Make the model accept batch as an argument parameter.
        output = self.model(*batch)
        return output


class MartAttack:
    """A minimal wrapper to run PyTorch-based MART adversary in Armory against PyTorch-based
    models.

    1. Extract the PyTorch model from an ART Estimator;
    2. Convert np.ndarray to torch.Tensor;
    3. Run PyTorch-based MART adversary and get result as torch.Tensor;
    4. Convert torch.Tensor back to np.ndarray.
    """

    def __init__(self, model, mart_adv_config_yaml):
        # Instantiate a MART adversary.
        adv_cfg = OmegaConf.load(mart_adv_config_yaml)
        adv = hydra.utils.instantiate(adv_cfg)

        self.batch_converter = adv.batch_converter
        self.adversary = adv.attack
        self.model_wrapper = adv.model_wrapper

        self.device = model.device
        self.model = self.model_wrapper(model)

        # Move adversary to the same device.
        self.adversary.to(self.device)

    def convert_batch_armory_to_torchvision(self, batch_armory_np):
        # np.ndarray -> torch.Tensor, on a device.
        # armory format -> torchvision format.
        batch_tv_pth = self.batch_converter(batch_armory_np, device=self.device)
        return batch_tv_pth

    def convert_batch_torchvision_to_armory(self, batch_tv_pth):
        # torchvision format -> armory format.
        # Note: revert(input, target)
        # torch.Tensor -> np.ndarray
        batch_armory_np = self.batch_converter.revert(*batch_tv_pth)
        return batch_armory_np

    def generate(self, **batch_armory_np):
        batch_tv_pth = self.convert_batch_armory_to_torchvision(batch_armory_np)

        batch_adv_tv_pth = self.adversary(batch=batch_tv_pth, model=self.model)
        batch_adv_armory_np = self.convert_batch_torchvision_to_armory(batch_adv_tv_pth)

        # Only return adversarial input in the original numpy format.
        input_key = self.batch_converter.input_key
        input_adv_np = batch_adv_armory_np[input_key]
        return input_adv_np
