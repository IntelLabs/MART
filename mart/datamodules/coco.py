#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import os
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import default_collate
from torchvision.datasets.coco import CocoDetection as CocoDetection_
from torchvision.datasets.folder import default_loader
from yolov3.datasets.utils import collate_img_label_fn as collate_img_label_fn_

__all__ = ["CocoDetection"]


class CocoDetection(CocoDetection_):
    """Extra features:
        1. Add image_id to the target dict;
        2. Add file_name to the target dict;
        3. Add ability to load multiple modalities

    Args:
        See torchvision.datasets.coco.CocoDetection

        modalities (list of strings): A list of subfolders under root to load modalities.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        modalities: Optional[List[str]] = None,
    ) -> None:
        # CocoDetection doesn't support transform or target_transform because
        # we need to manipulate the input and target at the same time.
        assert transform is None
        assert target_transform is None

        super().__init__(root, annFile, transform, target_transform, transforms)

        self.modalities = modalities

        # Targets can contain a lot of information...
        # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/4
        torch.multiprocessing.set_sharing_strategy("file_system")

    def _load_image(self, id: int) -> Any:
        if self.modalities is None:
            return super()._load_image(id)

        # Concatenate modalities into single tensor (not PIL.Image). We do this to be
        # compatible with existing transforms, since transforms should make the same
        # transformation to each modality.
        path = self.coco.loadImgs(id)[0]["file_name"]
        modalities = [
            default_loader(os.path.join(self.root, modality, path)) for modality in self.modalities
        ]

        # Create numpy.ndarry by stacking modalities along channels axis. We use numpy
        # because PIL does not support multi-channel images.
        image = np.concatenate([np.array(modality) for modality in modalities], axis=-1)

        return image

    def _load_target(self, id: int) -> List[Any]:
        annotations = super()._load_target(id)
        file_name = self.coco.loadImgs(id)[0]["file_name"]

        return {"image_id": id, "file_name": file_name, "annotations": annotations}

    def __getitem__(self, index: int):
        """Override __getitem__() to dictionarize input for multi-modality datasets.

        This runs after _load_image() and transforms(), while transforms() typically converts
        images to tensors.
        """

        image, target_dict = super().__getitem__(index)

        # Convert multi-modal input to a dictionary.
        if self.modalities is not None:
            # We assume image is a multi-channel tensor, with each modality including 3 channels.
            assert image.shape[0] == len(self.modalities) * 3
            image = dict(zip(self.modalities, image.split(3)))

        return image, target_dict


# Source: https://github.com/pytorch/vision/blob/dc07ac2add8285e16a716564867d0b4b953f6735/references/detection/utils.py#L203
def collate_fn(batch):
    return tuple(zip(*batch))


def to_padded_tensor(tensors, dim=0, fill=0.0):
    sizes = np.array([list(t.shape) for t in tensors])
    max_dim_size = sizes[:, dim].max()
    sizes[:, dim] = max_dim_size - sizes[:, dim]

    zeros = [
        torch.full(s.tolist(), fill, device=t.device, dtype=t.dtype)
        for t, s in zip(tensors, sizes)
    ]
    tensors = [torch.cat((t, z), dim=dim) for t, z in zip(tensors, zeros)]

    return tensors


def yolo_collate_fn(batch):
    images, targets = tuple(zip(*batch))

    images = default_collate(images)

    # Turn tuple of dicts into dict of tuples
    keys = targets[0].keys()
    target = {k: tuple(t[k] for t in targets) for k in keys}

    # Pad packed using torch.nested
    target["packed"] = to_padded_tensor(target["packed"])

    COLLATABLE_KEYS = ["packed", "packed_length", "perturbable_mask"]

    for key in target.keys():
        if key in COLLATABLE_KEYS:
            target[key] = default_collate(target[key])

    return images, target
