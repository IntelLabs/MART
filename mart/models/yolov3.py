#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch
import torch.nn.functional as F
import yolov3
from yolov3.inference import post_process
from yolov3.training import yolo_loss_fn
from yolov3.utils import cxcywh_to_xywh
from yolov3.model import YoloNetV3 as YoloNetV3_, YoloLayer as YoloLayer_
from yolov3.config import ANCHORS, NUM_ANCHORS_PER_SCALE, NUM_CLASSES, NUM_ATTRIB
from mart.utils import MonkeyPatch


class YoloNetV3(YoloNetV3_):
    def __init__(self):
        with MonkeyPatch(yolov3.model, "YoloLayer", YoloLayer):
            super().__init__()

    def forward(self, x):
        tmp1, tmp2, tmp3 = self.darknet(x)
        out1, out2, out3 = self.yolo_tail(tmp1, tmp2, tmp3)
        logits = torch.cat((out1["logits"], out2["logits"], out3["logits"]), 1)
        preds = torch.cat((out1["preds"], out2["preds"], out3["preds"]), 1)

        return {"logits": logits, "preds": preds}

class YoloLayer(torch.nn.Module):
    def __init__(self, scale, stride):
        super().__init__()

        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            idx = None

        self.anchors = torch.tensor([ANCHORS[i] for i in idx])
        self.stride = stride

    def forward(self, x):
        num_batch = x.size(0)
        num_grid = x.size(2)

        output_raw = x.view(num_batch,
                            NUM_ANCHORS_PER_SCALE,
                            NUM_ATTRIB,
                            num_grid,
                            num_grid).permute(0, 1, 3, 4, 2).contiguous().view(num_batch, -1, NUM_ATTRIB)

        prediction_raw = x.view(num_batch,
                                NUM_ANCHORS_PER_SCALE,
                                NUM_ATTRIB,
                                num_grid,
                                num_grid).permute(0, 1, 3, 4, 2).contiguous()

        self.anchors = self.anchors.to(x.device).float()
        # Calculate offsets for each grid
        grid_tensor = torch.arange(num_grid, dtype=torch.float, device=x.device).repeat(num_grid, 1)
        grid_x = grid_tensor.view([1, 1, num_grid, num_grid])
        grid_y = grid_tensor.t().view([1, 1, num_grid, num_grid])
        anchor_w = self.anchors[:, 0:1].view((1, -1, 1, 1))
        anchor_h = self.anchors[:, 1:2].view((1, -1, 1, 1))

        # Get outputs
        x_center_pred = (torch.sigmoid(prediction_raw[..., 0]) + grid_x) * self.stride # Center x
        y_center_pred = (torch.sigmoid(prediction_raw[..., 1]) + grid_y) * self.stride  # Center y
        w_pred = torch.exp(prediction_raw[..., 2]) * anchor_w  # Width
        h_pred = torch.exp(prediction_raw[..., 3]) * anchor_h  # Height
        bbox_pred = torch.stack((x_center_pred, y_center_pred, w_pred, h_pred), dim=4).view((num_batch, -1, 4)) #cxcywh
        conf_pred = torch.sigmoid(prediction_raw[..., 4]).view(num_batch, -1, 1)  # Conf
        cls_pred = torch.sigmoid(prediction_raw[..., 5:]).view(num_batch, -1, NUM_CLASSES)  # Cls pred one-hot.

        output = torch.cat((bbox_pred, conf_pred, cls_pred), -1)

        return {"logits": output_raw, "preds": output}


class Loss(torch.nn.Module):
    def __init__(self, image_size, average=True):
        super().__init__()

        self.image_size = image_size
        self.average = average

    def forward(self, logits, target, **kwargs):
        targets = target["target"]
        lengths = target["lengths"]

        losses = yolo_loss_fn(logits, targets, lengths, self.image_size, self.average)
        total_loss, coord_loss, obj_loss, noobj_loss, class_loss = losses

        pred_conf_logit = logits[..., 4]
        pred_conf_score = torch.sigmoid(pred_conf_logit)
        class_logits = logits[..., 5:]
        target_mask = (torch.argmax(class_logits, dim=-1) == 0) & (pred_conf_score > 0.1)

        # make objectness go to zero
        tgt_zero = torch.zeros(pred_conf_logit.size(), device=pred_conf_logit.device)
        hide_objects_losses = F.binary_cross_entropy_with_logits(
            pred_conf_logit, tgt_zero, reduction="none"
        )
        hide_objects_loss = hide_objects_losses.sum()

        # make target objectness go to zero
        hide_target_objects_loss = hide_objects_losses[target_mask].sum()

        # make target logit go to zero
        target_class_logit = class_logits[..., 0]  # 0 == person
        target_class_losses = F.binary_cross_entropy_with_logits(
            target_class_logit, tgt_zero, reduction="none"
        )
        target_class_loss = target_class_losses.sum()

        # make correctly predicted target class logit go to zero
        correct_target_class_loss = target_class_losses[target_mask].sum()

        return {
            "total_loss": total_loss,
            "coord_loss": coord_loss,
            "obj_loss": obj_loss,
            "noobj_loss": noobj_loss,
            "class_loss": class_loss,
            "hide_objects_loss": hide_objects_loss,
            "hide_target_objects_loss": hide_target_objects_loss,
            "target_class_loss": target_class_loss,
            "correct_target_class_loss": correct_target_class_loss,
            "target_count": target_mask.sum(),
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
    def forward(self, preds, target, **kwargs):
        detections = post_process(preds, self.nms, self.conf_thres, self.nms_thres)

        # FIXME: This should be another module
        # Convert detections and targets to List[dict[str, torch.Tensor]]. This is the format
        # torchmetrics wants.
        preds = [Detections.tensor_to_dict(det) for det in detections]

        targets = target["target"]
        lengths = target["lengths"]
        targets = [target[:length] for target, length in zip(targets, lengths)]
        targets = [Detections.tensor_to_dict(target) for target in targets]

        return {"preds": preds, "targets": targets}
