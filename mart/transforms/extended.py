#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

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
    "LoadTensors",
    "ConvertInstanceSegmentationToPerturbable",
    "RandomHorizontalFlip",
    "ConvertCocoPolysToMask",
    "PadToSquare",
    "Resize",
    "ConvertBoxesToCXCYWH",
    "RemapLabels",
    "PackBoxesAndLabels",
    "CreatePerturbableMaskFromImage",
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
        perturbable_mask = torch.sum(target["masks"], dim=0, keepdim=True) == 0
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


class LoadTensors(ExTransform):
    def __init__(self, root, ext=".pt") -> None:
        self.root = root
        self.ext = ext

    def __call__(self, image, target):
        filename, ext = os.path.splitext(target["file_name"])

        metadata = torch.load(
            os.path.join(self.root, filename + self.ext), map_location=image.device
        )
        assert isinstance(metadata, dict)

        for key in metadata:
            assert key not in target
            target[key] = metadata[key]

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
        self, image: Tensor, target: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
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


class PadToSquare(ExTransform):
    def __init__(self, fill):
        self.fill = fill

    def __call__(
        self,
        image: Tensor,  # CHW
        target: dict[str, Tensor] | None = None,
    ):
        w, h = F.get_image_size(image)

        l_or_t = abs(h - w) // 2
        r_or_b = abs(h - w) - l_or_t

        # padding is  (left, top, right, bottom)
        if h > w:
            padding = (l_or_t, 0, r_or_b, 0)
        else:
            padding = (0, l_or_t, 0, r_or_b)

        image = F.pad(image, padding, fill=self.fill)

        if target is not None:
            if "boxes" in target:
                target["boxes"] = self.pad_boxes(target["boxes"], padding)
            if "masks" in target:
                target["masks"] = self.pad_masks(target["masks"], padding)
            if "keypoints" in target:
                target["keypoints"] = self.pad_keypoints(target["keypoints"], padding)
            if "perturbable_mask" in target:
                target["perturbable_mask"] = self.pad_masks(target["perturbable_mask"], padding)
            if "gs_coords" in target:
                target["gs_coords"] = self.pad_coordinates(target["gs_coords"], padding)

        return image, target

    def pad_boxes(self, boxes, padding):
        boxes[:, 0] += padding[0]  # X + left
        boxes[:, 1] += padding[1]  # Y + top
        boxes[:, 2] += padding[0]  # X + left
        boxes[:, 3] += padding[1]  # Y + top

        return boxes

    def pad_masks(self, masks, padding):
        return F.pad(masks, padding, fill=0)

    def pad_keypoints(self, keypoints, padding):
        raise NotImplementedError

    def pad_coordinates(self, coordinates, padding):
        # coordinates are [[left, top], [right, top], [right, bottom], [left, bottom]]
        # padding is [left, top, right bottom]
        coordinates[:, 0] += padding[0]  # left padding
        coordinates[:, 1] += padding[1]  # top padding

        return coordinates


class Resize(ExTransform):
    def __init__(self, size):
        self.size = size

    def __call__(
        self,
        image: Tensor,
        target: dict[str, Tensor] | None = None,
    ):
        orig_w, orig_h = F.get_image_size(image)
        image = F.resize(image, size=self.size)
        new_w, new_h = F.get_image_size(image)

        dw, dh = new_w / orig_w, new_h / orig_h

        if target is not None:
            if "boxes" in target:
                target["boxes"] = self.resize_boxes(target["boxes"], (dw, dh))
            if "masks" in target:
                target["masks"] = self.resize_masks(target["masks"], (dw, dh))
            if "keypoints" in target:
                target["keypoints"] = self.resize_keypoints(target["keypoints"], (dw, dh))
            if "perturbable_mask" in target:
                target["perturbable_mask"] = self.resize_masks(
                    target["perturbable_mask"], (dw, dh)
                )
            if "gs_coords" in target:
                target["gs_coords"] = self.resize_coordinates(target["gs_coords"], (dw, dh))

        return image, target

    def resize_boxes(self, boxes, ratio):
        boxes[:, 0] *= ratio[0]  # X1 * width ratio
        boxes[:, 1] *= ratio[1]  # Y1 * height ratio
        boxes[:, 2] *= ratio[0]  # X2 * width ratio
        boxes[:, 3] *= ratio[1]  # Y2 * height ratio

        return boxes

    def resize_masks(self, masks, ratio):
        assert len(masks.shape) == 3

        # Resize fails on empty tensors
        if masks.shape[0] == 0:
            return torch.zeros((0, *self.size), dtype=masks.dtype, device=masks.device)

        return F.resize(masks, size=self.size, interpolation=F.InterpolationMode.NEAREST)

    def resize_keypoints(self, keypoints, ratio):
        raise NotImplementedError

    def resize_coordinates(self, coordinates, ratio):
        # coordinates are [[left, top], [right, top], [right, bottom], [left, bottom]]
        # ratio is [width, height]
        coordinates[:, 0] = (coordinates[:, 0] * ratio[0]).to(int)  # width ratio
        coordinates[:, 1] = (coordinates[:, 1] * ratio[1]).to(int)  # height ratio

        return coordinates


class ConvertBoxesToCXCYWH(ExTransform):
    def __call__(
        self,
        image: Tensor,
        target: dict[str, Tensor],
    ):
        # X1Y1X2Y2
        boxes = target["boxes"]

        # X2Y2 -> WH
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]

        # X1Y1 -> CXCY
        boxes[:, 0] += boxes[:, 2] / 2
        boxes[:, 1] += boxes[:, 3] / 2

        target["boxes"] = boxes

        return image, target


class RemapLabels(ExTransform):
    COCO_MAP = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6,
        8: 7,
        9: 8,
        10: 9,
        11: 10,
        13: 11,
        14: 12,
        15: 13,
        16: 14,
        17: 15,
        18: 16,
        19: 17,
        20: 18,
        21: 19,
        22: 20,
        23: 21,
        24: 22,
        25: 23,
        27: 24,
        28: 25,
        31: 26,
        32: 27,
        33: 28,
        34: 29,
        35: 30,
        36: 31,
        37: 32,
        38: 33,
        39: 34,
        40: 35,
        41: 36,
        42: 37,
        43: 38,
        44: 39,
        46: 40,
        47: 41,
        48: 42,
        49: 43,
        50: 44,
        51: 45,
        52: 46,
        53: 47,
        54: 48,
        55: 49,
        56: 50,
        57: 51,
        58: 52,
        59: 53,
        60: 54,
        61: 55,
        62: 56,
        63: 57,
        64: 58,
        65: 59,
        67: 60,
        70: 61,
        72: 62,
        73: 63,
        74: 64,
        75: 65,
        76: 66,
        77: 67,
        78: 68,
        79: 69,
        80: 70,
        81: 71,
        82: 72,
        84: 73,
        85: 74,
        86: 75,
        87: 76,
        88: 77,
        89: 78,
        90: 79,
    }

    def __init__(
        self,
        label_map: dict[int, int] | None = None,
    ):
        if label_map is None:
            label_map = self.COCO_MAP

        self.label_map = label_map

    def __call__(
        self,
        image: Tensor,
        target: dict[str, Tensor],
    ):
        labels = target["labels"]

        # This is a terrible implementation
        for i, label in enumerate(labels):
            labels[i] = self.label_map[label.item()]

        target["labels"] = labels

        return image, target


class PackBoxesAndLabels(ExTransform):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(
        self,
        image: Tensor,
        target: dict[str, Tensor],
    ):
        boxes = target["boxes"]
        labels = target["labels"]
        scores = torch.ones_like(labels)[..., None]

        labels = torch.nn.functional.one_hot(labels, num_classes=self.num_classes)

        target["packed"] = torch.cat([boxes, scores, labels], dim=-1)
        target["packed_length"] = target["packed"].shape[0]

        return image, target


class CreatePerturbableMaskFromImage(ExTransform):
    def __init__(self, chroma_key, threshold):
        self.chroma_key = torch.tensor(chroma_key)
        self.threshold = threshold

    def __call__(
        self,
        image: Tensor,
        target: dict[str, Tensor],
    ):
        self.chroma_key = self.chroma_key.to(image.device)

        l2_dist = ((image - self.chroma_key[:, None, None]) ** 2).sum(dim=0, keepdim=True).sqrt()
        perturbable_mask = l2_dist <= self.threshold

        target["perturbable_mask"] = perturbable_mask.float()

        return image, target
