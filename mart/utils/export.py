#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import json
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchvision.ops import box_convert

__all__ = ["CocoPredictionJSON"]


class CocoPredictionJSON(Metric):
    """Export prediction.json for object detection models."""

    def __init__(
        self,
        prediction_file_name: str = None,
        groundtruth_file_name: str = None,
        **kwargs,
    ) -> None:  # type: ignore
        super().__init__(**kwargs)

        self.prediction_file_name = prediction_file_name
        self.groundtruth_file_name = groundtruth_file_name

        self.add_state("detection_boxes", default=[], dist_reduce_fx=None)
        self.add_state("detection_scores", default=[], dist_reduce_fx=None)
        self.add_state("detection_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_boxes", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)
        self.add_state("image_id_list", default=[], dist_reduce_fx=None)

    def update(self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]) -> None:  # type: ignore
        """Add detections and groundtruth to the metric.

        The code is adapted from https://github.com/PyTorchLightning/metrics/blob/v0.6.0/torchmetrics/detection/map.py#L229
            Add image_id_list.

        Args:
            preds: A list consisting of dictionaries each containing the key-values\
            (each dictionary corresponds to a single image):
            - ``boxes``: torch.FloatTensor of shape
                [num_boxes, 4] containing `num_boxes` detection boxes of the format
                [xmin, ymin, xmax, ymax] in absolute image coordinates.
            - ``scores``: torch.FloatTensor of shape
                [num_boxes] containing detection scores for the boxes.
            - ``labels``: torch.IntTensor of shape
                [num_boxes] containing 0-indexed detection classes for the boxes.

            target: A list consisting of dictionaries each containing the key-values\
            (each dictionary corresponds to a single image):
            - ``boxes``: torch.FloatTensor of shape
                [num_boxes, 4] containing `num_boxes` groundtruth boxes of the format
                [xmin, ymin, xmax, ymax] in absolute image coordinates.
            - ``labels``: torch.IntTensor of shape
                [num_boxes] containing 1-indexed groundtruth classes for the boxes.

        Raises:
            ValueError:
                If ``preds`` is not of type List[Dict[str, torch.Tensor]]
            ValueError:
                If ``target`` is not of type List[Dict[str, torch.Tensor]]
            ValueError:
                If ``preds`` and ``target`` are not of the same length
            ValueError:
                If any of ``preds.boxes``, ``preds.scores``
                and ``preds.labels`` are not of the same length
            ValueError:
                If any of ``target.boxes`` and ``target.labels`` are not of the same length
            ValueError:
                If any box is not type float and of length 4
            ValueError:
                If any class is not type int and of length 1
            ValueError:
                If any score is not type float and of length 1
        """

        for item in preds:
            self.detection_boxes.append(item["boxes"])
            self.detection_scores.append(item["scores"])
            self.detection_labels.append(item["labels"])

        for item in target:
            self.groundtruth_boxes.append(item["boxes"])
            self.groundtruth_labels.append(item["labels"])
            self.image_id_list.append(item["image_id"])

    def compute(self) -> dict:
        annotations = _get_coco_format_annotations(
            self.detection_boxes,
            self.detection_labels,
            scores=self.detection_scores,
            image_id_list=self.image_id_list,
        )
        json.dump(annotations, open(self.prediction_file_name, "w"))

        groundtruth = _get_coco_format_annotations(
            self.groundtruth_boxes,
            self.groundtruth_labels,
            scores=None,
            image_id_list=self.image_id_list,
        )
        json.dump(groundtruth, open(self.groundtruth_file_name, "w"))

        return torch.tensor(0.0)


def _get_coco_format_annotations(
    boxes: List[torch.Tensor],
    labels: List[torch.Tensor],
    scores: Optional[List[torch.Tensor]] = None,
    image_id_list=None,
) -> Dict:
    """Transforms and returns all cached targets or predictions in COCO format.

    Format is defined at https://cocodataset.org/#format-data

    The code is adapted from
    https://github.com/PyTorchLightning/metrics/blob/v0.6.0/torchmetrics/detection/map.py#L356
    Return annotations only;
    Add image_id.
    """

    annotations = []
    annotation_id = 1  # has to start with 1, otherwise COCOEval results are wrong

    boxes = [box_convert(box, in_fmt="xyxy", out_fmt="xywh") for box in boxes]
    for i, (image_id, image_boxes, image_labels) in enumerate(zip(image_id_list, boxes, labels)):
        image_boxes = image_boxes.cpu().tolist()
        image_labels = image_labels.cpu().tolist()
        image_id = image_id.cpu().item()

        for k, (image_box, image_label) in enumerate(zip(image_boxes, image_labels)):
            if len(image_box) != 4:
                raise ValueError(
                    f"Invalid input box of sample {image_id}, element {k} (expected 4 values, got {len(image_box)})"
                )

            if not isinstance(image_label, int):
                raise ValueError(
                    f"Invalid input class of sample {image_id}, element {k}"
                    f" (expected value of type integer, got type {type(image_label)})"
                )

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "bbox": image_box,
                "category_id": image_label,
            }
            if scores is not None:
                score = scores[i][k].cpu().tolist()
                if not isinstance(score, float):
                    raise ValueError(
                        f"Invalid input score of sample {image_id}, element {k}"
                        f" (expected value of type float, got type {type(score)})"
                    )
                annotation["score"] = score
            annotations.append(annotation)
            annotation_id += 1

    return annotations
