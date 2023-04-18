#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch
from yolox.utils import postprocess


class Detections(torch.nn.Module):
    def __init__(self, num_classes=80, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        super().__init__()

        self.num_classes = num_classes
        self.conf_thre = conf_thre
        self.nms_thre = nms_thre
        self.class_agnostic = class_agnostic

    @staticmethod
    def cxcywh2xyxy(bboxes):
        bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
        bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        return bboxes

    @staticmethod
    def tensor_to_dict(detection):
        if detection is None:
            # Handle images with no detections
            boxes = torch.empty((0, 4), device="cuda") # HACK
            labels = torch.empty((0,), device="cuda") # HACK
            scores = torch.empty((0,), device="cuda") # HACK

        elif detection.shape[1] > 5:
            # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            boxes = detection[:, 0:4]
            labels = detection[:, 6].to(int)
            scores = detection[:, 4] * detection[:, 5]

        else:  # targets have no scores
            # [class, xc, yc, w, h]
            boxes = detection[:, 1:5]
            boxes = Detections.cxcywh2xyxy(boxes)
            labels = detection[:, 0].to(int)
            scores = torch.ones_like(labels)

            length = (labels > 0).sum()

            boxes = boxes[:length]
            labels = labels[:length]
            scores = scores[:length]

        return {"boxes": boxes, "labels": labels, "scores": scores}

    def forward(self, predictions, targets):
        detections = postprocess(
            predictions,
            self.num_classes,
            conf_thre=self.conf_thre,
            nms_thre=self.nms_thre,
            class_agnostic=self.class_agnostic,
        )

        # Convert preds and targets to format acceptable to torchmetrics
        preds = [Detections.tensor_to_dict(det) for det in detections]
        targets = [Detections.tensor_to_dict(tar) for tar in targets]

        return {"preds": preds, "target": targets}
