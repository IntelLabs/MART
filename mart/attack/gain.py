#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from typing import Optional

import torch

__all__ = ["Gain"]


class Gain(torch.nn.Module):
    """Gain functions must be differentiable so we inherit from nn.Module."""

    pass


class RoIHeadTargetClass(Gain):
    """The gain function encourages logits being classified as a particular class, e.g. background
    (class_index==0 in RCNN)."""

    def __init__(self, class_index: Optional[int] = 0, targeted: Optional[bool] = True) -> None:
        super().__init__()

        self.gain = torch.nn.CrossEntropyLoss()
        self.class_index = class_index
        self.targeted = targeted

    def forward(
        self, roi_heads_class_logits: torch.Tensor, proposals: torch.Tensor
    ) -> torch.Tensor:
        """

        Args:
            roi_heads_class_logits (torch.Tensor): Class logits from roi_heads.
            proposals (_type_): We only want to know how many proposals are there for one input.

        Returns:
            torch.Tensor: A gain vector with separate gain value for each input.
        """
        target = [self.class_index] * len(roi_heads_class_logits)
        device = roi_heads_class_logits.device
        target = torch.tensor(target, device=device)

        # Split class logits by input.
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        roi_heads_class_logits_list = roi_heads_class_logits.split(boxes_per_image, 0)
        target_list = target.split(boxes_per_image, 0)

        gains = []
        for batch_logits, batch_target in zip(roi_heads_class_logits_list, target_list):
            gain = self.gain(batch_logits, batch_target)
            if self.targeted:
                gain = -gain
            gains.append(gain)

        gains = torch.stack(gains)

        return gains


class RegionProposalScore(Gain):
    """The gain function to encourage background or foreground in region proposals.

    rpn_objectness is the sigmoid input. The lower value, the more likely to be background.
    """

    def __init__(self, background: Optional[bool] = True) -> None:
        """"""
        super().__init__()

        self.background = background

    def forward(self, rpn_objectness: torch.Tensor) -> torch.Tensor:
        logits = torch.cat([logits.reshape(-1) for logits in rpn_objectness])
        # TODO: We may remove sigmoid.
        probs = torch.sigmoid(logits)
        # prob_mean = probs.mean()
        if self.background:
            # Encourage background.
            return -probs
        else:
            # Encourage foreground.
            return probs
