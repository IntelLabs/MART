#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from functools import reduce

import torch

from mart.attack.batch_converter import BatchConverter


class ObjectDetectionBatchConverter(BatchConverter):
    def __init__(
        self,
        input_key: str = "x",
        target_keys: dict = {
            "y": ["area", "boxes", "id", "image_id", "is_crowd", "labels"],
            "y_patch_metadata": [
                "avg_patch_depth",
                "gs_coords",
                "mask",
                "max_depth_perturb_meters",
            ],
        },
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_key = input_key
        self.target_keys = target_keys

    def _convert(self, batch: dict):
        input = batch[self.input_key]

        target = []
        all_targets = [batch[key] for key in self.target_keys]

        # Merge several target keys.
        for dicts in zip(*all_targets):
            joint_target = reduce(lambda a, b: a | b, dicts)
            target.append(joint_target)

        # NHWC -> NCHW, the PyTorch format.
        input = input.permute((0, 3, 1, 2))
        # NCHW -> tuple[CHW]
        input = tuple(inp_ for inp_ in input)

        target = tuple(target)

        return input, target

    def _revert(self, input: tuple[torch.Tensor], target: tuple[dict]) -> dict:
        batch = {}

        # tuple[CHW] -> NCHW
        input = torch.stack(input)
        # NCHW -> NHWC, the TensorFlow format used in ART.
        input = input.permute((0, 2, 3, 1))

        batch[self.input_key] = input

        # Split target into several self.target_keys
        for target_key, sub_keys in self.target_keys.items():
            batch[target_key] = []
            for target_i_dict in target:
                target_key_i = {sub_key: target_i_dict[sub_key] for sub_key in sub_keys}
                batch[target_key].append(target_key_i)

        return batch
