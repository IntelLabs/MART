#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import hydra
from omegaconf import OmegaConf


class MartAttack:
    """A minimal wrapper to run MART adversary in Armory against PyTorch-based models.

    1. Instantiate an adversary that runs attack in MART;
    2. Instantiate batch_converter that turns Armory's numpy batch into the PyTorch batch;
    3. The adversary.model_transform() extracts the PyTorch model from an ART Estimator and makes other changes to easier attack;
    4. The adversary returns adversarial examples in the PyTorch format;
    5. The batch_converter reverts the adversarial examples into the numpy format.
    """

    def __init__(self, model, mart_adv_config_yaml):
        """_summary_

        Args:
            model (Callable): An ART Estimator that contains a PyTorch model.
            mart_adv_config_yaml (str): File path to the adversary configuration.
        """
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
        # Armory format -> torchvision format
        batch_tv_pth = self.batch_converter(batch_armory_np, device=self.device)
        # Attack
        batch_adv_tv_pth = self.adversary(batch=batch_tv_pth, model=self.model)
        # torchvision format -> Armory format
        batch_adv_armory_np = self.batch_converter.revert(*batch_adv_tv_pth)

        # Only return adversarial input in the original numpy format.
        input_key = self.batch_converter.input_key
        input_adv_np = batch_adv_armory_np[input_key]
        return input_adv_np
