#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

from mart.utils.monkey_patch import MonkeyPatch

__all__ = ["DualMode", "DualModeGeneralizedRCNN"]


class DualMode(torch.nn.Module):
    """Run model.forward() in both the training mode and the eval mode, then aggregate results in a
    dictionary {"training": ..., "eval": ...}.

    Some object detection models are implemented to return losses in the training mode and
    predictions in the eval mode, but we want both the losses and the predictions when attacking a
    model in the test mode.
    """

    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, *args, **kwargs):
        original_training_status = self.model.training
        ret = {}

        # TODO: Reuse the feature map in dual mode to improve efficiency
        self.model.train(True)
        ret["training"] = self.model(*args, **kwargs)

        self.model.train(False)
        with torch.no_grad():
            ret["eval"] = self.model(*args, **kwargs)

        self.model.train(original_training_status)
        return ret


class DualModeGeneralizedRCNN(torch.nn.Module):
    """Efficient dual mode for GeneralizedRCNN from torchvision, by reusing feature maps from
    backbone."""

    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, *args, **kwargs):
        bound_method = self.forward_dual_mode.__get__(self.model, self.model.__class__)
        with MonkeyPatch(self.model, "forward", bound_method):
            ret = self.model(*args, **kwargs)
        return ret

    # Adapted from: https://github.com/pytorch/vision/blob/32757a260dfedebf71eb470bd0a072ed20beddc3/torchvision/models/detection/generalized_rcnn.py#L46
    @staticmethod
    def forward_dual_mode(self, images, targets=None):
        # type: (GeneralizedRCNN, List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            self (GeneralizedRCNN): the model.
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # Validate targets in both training and eval mode.
        if targets is not None:
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                    )
                else:
                    torch._assert(
                        False,
                        f"Expected target boxes to be of type Tensor, got {type(boxes)}.",
                    )

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        original_training_status = self.training
        ret = {}

        # Training mode.
        self.train(True)
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        ret["training"] = losses

        # Eval mode.
        self.train(False)
        with torch.no_grad():
            proposals, proposal_losses = self.rpn(images, features, targets)
            detections, detector_losses = self.roi_heads(
                features, proposals, images.image_sizes, targets
            )
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
            ret["eval"] = detections

        self.train(original_training_status)

        return ret
