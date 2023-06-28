#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models.detection.transform import (
    paste_masks_in_image,
    resize_boxes,
    resize_keypoints,
)

__all__ = ["GeneralizedRCNNPostProcessor"]


class GeneralizedRCNNPostProcessor(nn.Module):
    def forward(
        self,
        result: list[dict[str, Tensor]],
        image_shapes: list[tuple[int, int]],
        original_images: list[Tensor],
    ) -> list[dict[str, Tensor]]:
        original_image_sizes = get_image_sizes(original_images)

        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result


def get_image_sizes(images: list[Tensor]):
    image_sizes: list[tuple[int, int]] = []

    for img in images:
        val = img.shape[-2:]
        torch._assert(
            len(val) == 2,
            f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        )
        image_sizes.append((val[0], val[1]))

    return image_sizes
