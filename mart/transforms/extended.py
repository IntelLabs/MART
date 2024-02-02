#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

# FIXME: We really shouldn't be importing private functions
from .torchvision_ref import ConvertCocoPolysToMask as ConvertCocoPolysToMask_
from .torchvision_ref import _flip_coco_person_keypoints

logger = logging.getLogger(__name__)

__all__ = [
    "ExTransform",
    "Compose",
    "Lambda",
    "SplitLambda",
    "LoadPerturbableMask",
    "LoadCoords",
    "ConvertInstanceSegmentationToPerturbable",
    "RandomHorizontalFlip",
    "ConvertCocoPolysToMask",
]


class ExTransform:
    """Extended transforms support kwargs in __call__.

    This is useful for transforms that need to transform the target in addition to the input like
    with object detection models.
    """

    pass


class Compose(T.Compose, ExTransform):
    """Add supports for composing vanilla and extended Transforms.

    Args:
        transforms (list): A list of transforms to compose together
        return_kwargs (bool): Whether to return a dict or just the
            transformed input (default: False)
    """

    def __init__(self, transforms, return_kwargs=False):
        super().__init__(transforms)

        self.return_kwargs = return_kwargs

    def __call__(self, x_or_dict, target=None, **kwargs):
        kwargs["target"] = target
        if isinstance(x_or_dict, dict):
            x = x_or_dict.pop("x")
            kwargs = {**kwargs, **x_or_dict}
        else:
            x = x_or_dict

        for transform in self.transforms:
            if transform is None:
                continue

            if isinstance(transform, ExTransform):
                x = transform(x, **kwargs)
            else:
                x = transform(x)

            # Assume output is (x, target), adding target to kwargs
            if isinstance(x, tuple):
                x, new_target = x
                kwargs["target"] = new_target

        if self.return_kwargs:
            kwargs["x"] = x
            return kwargs
        elif target is not None:
            return x, kwargs["target"]
        else:
            return x


class Lambda(T.Lambda, ExTransform):
    def __init__(self, lambd):
        super().__init__(lambd)

    def __call__(self, tensor, **kwargs):
        return self.lambd(tensor, **kwargs)


class SplitLambda(Lambda):
    def __init__(self, lambd, split_size_or_sections, lambd_section=0, dim=0):
        super().__init__(lambd)

        self.split_size_or_sections = split_size_or_sections
        self.lambd_section = lambd_section
        self.dim = dim

    def __call__(self, tensor, **kwargs):
        sections = list(torch.split(tensor, self.split_size_or_sections, dim=self.dim))
        sections[self.lambd_section] = super().__call__(sections[self.lambd_section], **kwargs)
        tensor = torch.cat(sections, self.dim)

        return tensor


class ConvertInstanceSegmentationToPerturbable(ExTransform):
    """Merge all instance masks and reverse."""

    def __call__(self, image, target):
        perturbable_mask = torch.sum(target["masks"], dim=0) == 0
        # Convert to float to be differentiable.
        target["perturbable_mask"] = perturbable_mask.float()

        return image, target


class LoadPerturbableMask(ExTransform):
    """Load perturbable masks and add to target."""

    def __init__(self, perturb_mask_folder) -> None:
        self.perturb_mask_folder = perturb_mask_folder
        self.to_tensor = T.ToTensor()

    def __call__(self, image, target):
        im = Image.open(os.path.join(self.perturb_mask_folder, target["file_name"]))
        # The mask is in single channel, but the file is in RGB.
        im = ImageOps.grayscale(im)
        perturbable_mask = self.to_tensor(im)[0]
        # Convert to float to be differentiable.
        target["perturbable_mask"] = perturbable_mask
        return image, target


class LoadCoords(ExTransform):
    """Load perturbable masks and add to target."""

    def __init__(self, folder) -> None:
        self.folder = folder
        self.to_tensor = T.ToTensor()

    def __call__(self, image, target):
        file_name = os.path.splitext(target["file_name"])[0]
        coords_fname = f"{file_name}_coords.npy"
        coords_fpath = os.path.join(self.folder, coords_fname)
        coords = np.load(coords_fpath)

        coords = self.to_tensor(coords)[0]
        # Convert to float to be differentiable.
        target["coords"] = coords
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
