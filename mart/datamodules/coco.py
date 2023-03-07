#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import os
from typing import Any, Callable, List, Optional

import numpy as np
from torchvision.datasets.coco import CocoDetection as CocoDetection_
from torchvision.datasets.folder import default_loader

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
