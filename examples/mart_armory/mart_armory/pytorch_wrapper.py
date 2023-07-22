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

from mart.models.dual_mode import DualModeGeneralizedRCNN

from .batch_converter import ObjectDetectionBatchConverter


# A recursive function to convert all np.ndarray in an object to torch.Tensor, or vice versa.
@multimethod
def convert(obj: dict, device=None):
    return {key: convert(value, device=device) for key, value in obj.items()}


@multimethod
def convert(obj: list, device=None):  # noqa: F811
    return [convert(item, device=device) for item in obj]


@multimethod
def convert(obj: tuple, device=None):  # noqa: F811
    return tuple(convert(obj, device=device))


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


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()

        # FIXME: We need an interface to modify the model, because the model only returns prediction in eval() model.
        self.model = DualModeGeneralizedRCNN(model)

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

    def __init__(self, model, batch_converter, mart_adv_config_yaml):
        # Extract PyTorch model from an ART Estimator.
        # TODO: Automatically search for torch.nn.Module within model.
        self.model = ModelWrapper(model._model)
        self.device = model.device

        self.batch_converter = batch_converter

        # Instantiate a MART adversary.
        adv_cfg = OmegaConf.load(mart_adv_config_yaml)
        self.adversary = hydra.utils.instantiate(adv_cfg)

        # Move adversary to the same device.
        self.adversary.to(self.device)

    def convert_batch_armory_to_torchvision(self, batch_armory_np):
        # np.ndarray -> torch.Tensor, on a device.
        batch_armory_pth = convert(batch_armory_np, device=self.device)
        # armory format -> torchvision format.
        batch_tv_pth = self.batch_converter(batch_armory_pth)
        return batch_tv_pth

    def convert_batch_torchvision_to_armory(self, batch_tv_pth):
        # torchvision format -> armory format.
        # Note: revert(input, target)
        batch_armory_pth = self.batch_converter.revert(*batch_tv_pth)
        # torch.Tensor -> np.ndarray
        batch_armory_np = convert(batch_armory_pth)
        return batch_armory_np

    def generate(self, **batch_armory_np):
        batch_tv_pth = self.convert_batch_armory_to_torchvision(batch_armory_np)

        # FIXME: Convert perturbable_mask somewhere else.
        batch_tv_pth[1][0]["perturbable_mask"] = (
            batch_tv_pth[1][0]["mask"].permute((2, 0, 1)) / 255
        )

        batch_adv_tv_pth = self.adversary(batch=batch_tv_pth, model=self.model)
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
