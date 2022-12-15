#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import abc
from typing import List, Optional, Tuple, Union

import torch
from torchvision.ops import box_iou

from .base import Objective

__all__ = ["ZeroAP", "Missed"]


class ZeroAP(Objective):
    """Determine if predictions yields zero Average Precision."""

    def __init__(
        self,
        iou_threshold: Optional[Union[int, float]] = 0.5,
        confidence_threshold: Optional[Union[int, float]] = 0.5,
    ) -> None:
        super().__init__()

        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

    def __call__(
        self, preds: Union[torch.Tensor, List], target: Union[torch.Tensor, List, Tuple]
    ) -> torch.Tensor:
        # For each class in target,
        #   if there is one pred_box has IoU with one of the gt_box larger than iou_threshold
        #       return False
        # Return True
        achieved_list = []

        for pred, gt in zip(preds, target):
            achieved = True
            for gt_cls in set(gt["labels"].cpu().numpy()):
                # Ground truth boxes with the same class.
                gt_boxes = gt["boxes"][gt["labels"] == gt_cls]
                # The same class and confident enough prediction
                pred_boxes_idx = torch.logical_and(
                    pred["labels"] == gt_cls,
                    pred["scores"] >= self.confidence_threshold,
                )
                pred_boxes = pred["boxes"][pred_boxes_idx]

                iou_pairs = box_iou(gt_boxes, pred_boxes)
                if iou_pairs.numel() > 0 and iou_pairs.max().item() >= self.iou_threshold:
                    achieved = False
                    break

            achieved_list.append(achieved)

        device = target[0]["boxes"].device
        achieved_tensor = torch.tensor(achieved_list, device=device)

        return achieved_tensor


class Missed(Objective):
    """The objective of the adversary is to make all AP errors as the missed error, i.e. no object
    is detected, nor false positive."""

    def __init__(self, confidence_threshold: Optional[Union[int, float]] = 0.5) -> None:
        super().__init__()

        self.confidence_threshold = confidence_threshold

    def __call__(
        self, preds: Union[torch.Tensor, List], target: Union[torch.Tensor, List, Tuple]
    ) -> torch.Tensor:
        achieved_list = []

        for pred in preds:
            if (pred["scores"] >= self.confidence_threshold).sum().item() > 0:
                achieved_list.append(False)
            else:
                achieved_list.append(True)

        device = preds[0]["boxes"].device
        achieved_tensor = torch.tensor(achieved_list, device=device)

        return achieved_tensor
