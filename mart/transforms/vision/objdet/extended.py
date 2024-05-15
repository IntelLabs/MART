#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import transforms as T

logger = logging.getLogger(__name__)

__all__ = [
    "ExTransform",
    "Compose",
    "Lambda",
    "SplitLambda",
    "LoadPerturbableMask",
    "LoadCoords",
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
