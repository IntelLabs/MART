#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import hydra
from omegaconf import OmegaConf


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

        self.model = model
        self.device = model.device

        # Move adversary to the same device.
        self.adversary.to(self.device)

    def generate(self, **batch_armory_np):
        # np.ndarray -> torch.Tensor, on a device.
        # armory format -> torchvision format.
        batch_tv_pth = self.batch_converter(batch_armory_np, device=self.device)

        batch_adv_tv_pth = self.adversary(batch=batch_tv_pth, model=self.model)

        # torchvision format -> armory format.
        # torch.Tensor -> np.ndarray
        batch_adv_armory_np = self.batch_converter.revert(*batch_adv_tv_pth)

        # Only return adversarial input in the original numpy format.
        input_key = self.batch_converter.input_key
        input_adv_np = batch_adv_armory_np[input_key]
        return input_adv_np
