#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

from .extended import ExTransform

# FIXME: We really shouldn't be importing private functions
from .torchvision_ref import ConvertCocoPolysToMask as ConvertCocoPolysToMask_
from .torchvision_ref import _flip_coco_person_keypoints

logger = logging.getLogger(__name__)

__all__ = [
    "ConvertInstanceSegmentationToPerturbable",
    "RandomHorizontalFlip",
    "ConvertCocoPolysToMask",
]


class ConvertInstanceSegmentationToPerturbable(ExTransform):
    """Merge all instance masks and reverse."""

    def __call__(self, image, target):
        perturbable_mask = torch.sum(target["masks"], dim=0) == 0
        # Convert to float to be differentiable.
        target["perturbable_mask"] = perturbable_mask.float()

        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip, ExTransform):
    """Flip the image and annotations including boxes, masks, keypoints and the
    perturable_masks."""

    @staticmethod
    def flip_boxes(image, target):
        width, _ = F.get_image_size(image)
        target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        return image, target

    @staticmethod
    def flip_masks(image, target):
        target["masks"] = target["masks"].flip(-1)
        return image, target

    @staticmethod
    def flip_keypoints(image, target):
        width, _ = F.get_image_size(image)
        keypoints = target["keypoints"]
        keypoints = _flip_coco_person_keypoints(keypoints, width)
        target["keypoints"] = keypoints
        return image, target

    @staticmethod
    def flip_perturable_masks(image, target):
        target["masks"] = target["masks"].flip(-1)
        return image, target

    @staticmethod
    def flip_perturbable_mask(image, target):
        target["perturbable_mask"] = target["perturbable_mask"].flip(-1)
        return image, target

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                image, target = self.flip_boxes(image, target)
                if "masks" in target:
                    image, target = self.flip_masks(image, target)
                if "keypoints" in target:
                    image, target = self.flip_keypoints(image, target)
                if "perturbable_mask" in target:
                    image, target = self.flip_perturable_masks(image, target)
        return image, target


class ConvertCocoPolysToMask(ConvertCocoPolysToMask_, ExTransform):
    pass
