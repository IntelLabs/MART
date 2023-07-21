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

from .batch_converter import ObjectDetectionBatchConverter


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

    def __init__(self, model, batch_converter, mart_adv_config_yaml):
        # TODO: Automatically search for torch.nn.Module within model.
        # Extract PyTorch model from an ART Estimator.
        self.model = model._model
        self.device = self.model.device

        self.batch_converter = batch_converter

        # Instantiate a MART adversary.
        adv_cfg = OmegaConf.load(mart_adv_config_yaml)
        self.adversary = hydra.utils.instantiate(adv_cfg)

    def convert_batch_armory_to_torchvision(self, batch_armory_np):
        # np.ndarray -> torch.Tensor, on a device.
        batch_armory_pth = convert(batch_armory_np, device=self.device)
        # armory format -> torchvision format.
        batch_tv_pth = self.batch_converter(batch_armory_pth)
        return batch_tv_pth

    def convert_batch_torchvision_to_armory(self, batch_tv_pth):
        # torchvision format -> armory format.
        batch_armory_pth = self.batch_converter.revert(batch_tv_pth)
        # torch.Tensor -> np.ndarray
        batch_armory_np = convert(batch_armory_pth)
        return batch_armory_np

    def generate(self, **batch_armory_np):
        batch_tv_pth = self.convert_batch_armory_to_torchvision(batch_armory_np)
        batch_adv_tv_pth = self.adversary(batch_tv_pth, model=self.model)
        batch_adv_armory_np = self.convert_batch_torchvision_to_armory(batch_adv_tv_pth)

        # Only return adversarial input in the original numpy format.
        input_key = self.batch_converter.input_key
        input_adv_np = batch_adv_armory_np[input_key]
        return input_adv_np


class MartAttackObjectDetection(MartAttack):
    def __init__(self, model, mart_adv_config_yaml):
        batch_config = {
            "input_key": "x",
            "target_keys": {
                "y": ["area", "boxes", "id", "image_id", "is_crowd", "labels"],
                "y_patch_metadata": [
                    "avg_patch_depth",
                    "gs_coords",
                    "mask",
                    "max_depth_perturb_meters",
                ],
            },
        }

        batch_converter = ObjectDetectionBatchConverter(**batch_config)
        super().__init__(
            model=model, batch_converter=batch_converter, mart_adv_config_yaml=mart_adv_config_yaml
        )
