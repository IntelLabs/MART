#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch
from yolov3.inference import post_process
from yolov3.training import yolo_loss_fn
from yolov3.utils import cxcywh_to_xywh


class Loss(torch.nn.Module):
    def __init__(self, image_size, average=True):
        super().__init__()

        self.image_size = image_size
        self.average = average

    def forward(self, logits, targets, target_lengths):
        losses = yolo_loss_fn(logits, targets, target_lengths, self.image_size, self.average)
        total_loss, coord_loss, obj_loss, noobj_loss, class_loss = losses

        return {
            "total_loss": total_loss,
            "coord_loss": coord_loss,
            "obj_loss": obj_loss,
            "noobj_loss": noobj_loss,
            "class_loss": class_loss,
        }


class Detections(torch.nn.Module):
    def __init__(self, nms=True, conf_thres=0.8, nms_thres=0.4):
        super().__init__()

        self.nms = nms
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

    @staticmethod
    def xywh_to_xyxy(boxes):
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        return boxes

    @staticmethod
    def tensor_to_dict(detection):
        boxes = detection[:, 0:4]
        scores = detection[:, 4]
        labels = detection[:, 5:]

        boxes = cxcywh_to_xywh(boxes)
        boxes = Detections.xywh_to_xyxy(boxes)

        if labels.shape[1] == 1:  # index
            labels = labels[:, 0].to(int)
        else:  # one-hot
            labels = labels.argmax(dim=1)

        return {"boxes": boxes, "labels": labels, "scores": scores}

    @torch.no_grad()
    def forward(self, logits, targets, target_lengths):
        detections = post_process(logits, self.nms, self.conf_thres, self.nms_thres)

        # Convert detections and targets to List[dict[str, torch.Tensor]]. This is the format
        # torchmetrics wants.
        preds = [Detections.tensor_to_dict(det) for det in detections]
        targets = [target[:length] for target, length in zip(targets, target_lengths)]
        targets = [Detections.tensor_to_dict(target) for target in targets]

        return {"preds": preds, "target": targets}
