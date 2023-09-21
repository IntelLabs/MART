#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset as VisionDataset_

logger = logging.getLogger(__name__)
try:
    # Disable the FiftyOne tracker by default due to the privacy concern.
    # Users need to export FIFTYONE_DO_NOT_TRACK=0 if they intend to be tracked.
    if os.getenv("FIFTYONE_DO_NOT_TRACK") is None:
        os.environ["FIFTYONE_DO_NOT_TRACK"] = "1"
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
        assert (
            dataset_name in fo_datasets
        ), f"Dataset {dataset_name} does not exist. To create a FiftyOne dataset, used the CLI command: https://docs.voxel51.com/cli/index.html#create-datasets"

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

        print(
            f"FiftyOne dataset {dataset_name}, with {len(self.ids)} samples, successfully loaded."
        )

    def __getitem__(self, index: int) -> Any:
        sample_id = self.ids[index]
        sample = self.filtered_dataset[sample_id]
        metadata = sample.metadata
        image_path = Path(sample.filepath)
        img = Image.open(image_path).convert("RGB")

        target = {}
        target["image_id"] = index
        target["file_name"] = image_path.name
        target["annotations"] = []

        detections = sample[self.gt_field].detections
        for det in detections:
            category_id = self.labels_map_rev[det.label]
            coco_obj = fouc.COCOObject.from_label(
                det,
                metadata,
                category_id=category_id,
            )

            coco_annotation = coco_obj.to_anno_dict()
            # If the detection object has segmentation information, verify that the
            # segmentation field is not empty
            if coco_obj.segmentation is not None and len(coco_annotation["segmentation"]) == 0:
                continue

            target["annotations"].append(coco_annotation)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.filtered_dataset)
