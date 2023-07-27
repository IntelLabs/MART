#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
import os
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset as VisionDataset_

logger = logging.getLogger(__name__)
try:
    import fiftyone as fo
    import fiftyone.utils.coco as fouc
except ImportError:
    logger.debug("fiftyone module is not installed!")

__all__ = ["FiftyOneDataset"]


class FiftyOneDataset(VisionDataset_):
    # Adapted from FiftyOne example: https://github.com/voxel51/fiftyone-examples/blob/master/examples/pytorch_detection_training.ipynb

    def __init__(
        self,
        dataset_name: str,
        gt_field: str,
        sample_tags: List[str] = [],
        label_tags: List[str] = [],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__("", transforms, transform, target_transform)

        self.gt_field = gt_field

        # Verify the FiftyOne
        fo_datasets = fo.list_datasets()
        assert dataset_name in fo_datasets, f"Dataset {dataset_name} does not exist!"

        # Load FiftyOne dataset
        self.dataset = fo.load_dataset(dataset_name)
        self.dataset = self.dataset.exists(self.gt_field)

        # filter with tags
        self.filtered_dataset = (
            self.dataset.match_tags(sample_tags) if len(sample_tags) > 0 else self.dataset
        )
        self.filtered_dataset = (
            self.filtered_dataset.select_labels(tags=label_tags)
            if len(label_tags) > 0
            else self.filtered_dataset
        )

        # extract samples' IDs
        self.ids = self.filtered_dataset.values("id")

        # set classes
        self.classes = self.filtered_dataset.default_classes
        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

        self.filtered_dataset.shuffle()
        print(
            f"FiftyOne dataset {dataset_name}, with {len(self.ids)} samples, successfully loaded."
        )

    def __getitem__(self, index: int) -> Any:
        sample_id = self.ids[index]
        sample = self.filtered_dataset[sample_id]
        metadata = sample.metadata
        img = Image.open(sample.filepath).convert("RGB")

        boxes = []
        labels = []
        area = []
        iscrowd = []
        detections = sample[self.gt_field].detections
        for det in detections:
            category_id = self.labels_map_rev[det.label]
            coco_obj = fouc.COCOObject.from_label(
                det,
                metadata,
                category_id=category_id,
            )
            x, y, w, h = coco_obj.bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(coco_obj.category_id)
            area.append(coco_obj.area)
            iscrowd.append(coco_obj.iscrowd)

        target = {}
        target["image_id"] = torch.as_tensor([index])
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.filtered_dataset)
