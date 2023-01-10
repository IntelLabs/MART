#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from typing import Any, List, Optional

import hydra
import numpy
import torch
from art.estimators.object_detection.object_detector import ObjectDetector
from omegaconf import OmegaConf

from ..nn import SequentialDict

__all__ = ["MartToArtAttackAdapter"]


class MartToArtAttackAdapter:
    """A wrapper to run MART attacks in ART.

    The wrapper is currently hard-coded for torchvision.models.detection.faster_rcnn.FasterRCNN,
    but it should be reusable for other models with slight modifications.
    """

    def __init__(self, target_model: ObjectDetector, mart_exp_config_yaml: str, **kwargs):
        """Run MART attacks in ART.

        Args:
            target_model (<class 'art.estimators.object_detection.object_detector.ObjectDetector'>): The target model to attack.
            mart_exp_config_yaml (str): File path to the yaml configuration of MART experiments.
        """
        exp_cfg = OmegaConf.load(mart_exp_config_yaml)

        modules = hydra.utils.instantiate(exp_cfg.model.modules)
        # Replace torchvision.models.detection.faster_rcnn.FasterRCNN with that in the target model.
        modules.losses_and_detections.model = target_model.model
        sequences = {"test": exp_cfg.model.test_sequence}
        model = SequentialDict(modules, sequences)

        # self.model is a MART-compatible SequentialDict model.
        self.model = model
        self.attack = model["input_adv_test"]

        self._device = target_model.device

    def generate(
        self,
        x: Optional[numpy.ndarray] = None,
        y: Optional[numpy.ndarray] = None,
        y_patch_metadata: Optional[List] = None,
    ):
        """ART invokes this method to produce x_adv.

        1. Convert ART's data `x`, `y` and `y_patch_metadata` to MART's data format `input` and `target`.
        3. Run the MART attack to get "input_adv";
        4. Convert MART's `input_adv` to `x_adv` in ART's data format.

        Args:
            x (numpy.ndarray): Image tensor in the NHWC order, N==1. Pixel values are between 0 and 1.
            y (list): List of targets, length is 1.
            y_patch_metadata (list): A one-length list of dictionaries with the perturbable patch information. The dictionary has three keys `['avg_patch_depth', 'gs_coords', 'mask']`, but we only use `mask`.

        Returns:
            numpy.ndarray: x with the adversarial perturbation.
        """

        input = self.convert_input_art_to_mart(x)
        target = self.convert_target_art_to_mart(y, y_patch_metadata)

        input_adv = self.attack(input, target, model=self.model, step="test")

        x_adv = self.convert_input_mart_to_art(input_adv)

        return x_adv

    def convert_input_art_to_mart(self, x: numpy.ndarray):
        """Convert ART input to the MART's format.

        Args:
            x (np.ndarray): NHWC, [0, 1]

        Returns:
            tuple: a tuple of tensors in CHW, [0, 255].
        """
        input = torch.tensor(x).permute((0, 3, 1, 2)).to(self._device) * 255
        input = tuple(inp_ for inp_ in input)
        return input

    def convert_input_mart_to_art(self, input: tuple):
        """Convert MART input to the ART's format.

        Args:
            input (tuple): a tuple of tensors in CHW, [0, 255].

        Returns:
            np.ndarray: NHWC, [0, 1]
        """
        x = torch.stack([inp_i.detach().permute((1, 2, 0)) / 255 for inp_i in input])
        x = x.cpu().numpy()
        return x

    def convert_target_art_to_mart(self, y: numpy.ndarray, y_patch_metadata: List):
        """Convert ART's target to the MART's format.
        1. np.ndarray -> torch.Tensor on self._device;
        2. Add "perturbable_mask" from `y_patch_metadata`;
        3. Add "file_name" from `image_id`.

        Args:
            y (_type_): _description_
            y_patch_metadata (_type_): _description_

        Returns:
            tuple: a tuple of target dictionaies.
        """
        # Copy y to target, and convert ndarray to pytorch tensors accordingly.
        target = []
        for yi, mi in zip(y, y_patch_metadata):
            target_i = {}
            for field in ["id", "labels", "boxes"]:
                target_i[field] = torch.tensor(yi[field]).to(self._device)

            # y_patch_metadata[0] has ['avg_patch_depth', 'gs_coords', 'mask'], but we only use 'mask'.
            # mask is in HWC, {0, 255}.
            # mask is mostly 0, with perturbable pixels as 255.
            # Convert to CHW {False, True}
            target_i["perturbable_mask"] = (
                torch.tensor(mi["mask"] > 0).permute((2, 0, 1)).to(self._device)
            )
            # Use image_id as the file_name because we don't have the file name.
            target_i["file_name"] = f"{yi['image_id'][0]}.jpg"
            target.append(target_i)

        target = tuple(target)

        return target
