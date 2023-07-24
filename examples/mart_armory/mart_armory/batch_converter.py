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

        target = tuple(target)

        # # TODO: Move to transform() that works on both input and target.
        # #   1. input permute
        # #   2. tuplize input
        # #   3. permute and scale target["mask"]
        # # NHWC -> NCHW, the PyTorch format.
        # input = input.permute((0, 3, 1, 2))
        # # NCHW -> tuple[CHW]
        # input = tuple(input)

        return input, target

    def _revert(self, input: tuple[torch.Tensor], target: tuple[dict]) -> dict:
        batch = {}

        # # TODO: Move to untransform().
        # #   1. permute and scale target["mask"]
        # #   2. input stack
        # #   3. input permute
        # # tuple[CHW] -> NCHW
        # input = torch.stack(input)
        # # NCHW -> NHWC, the TensorFlow format used in ART.
        # input = input.permute((0, 2, 3, 1))

        batch[self.input_key] = input

        # Split target into several self.target_keys
        for target_key, sub_keys in self.target_keys.items():
            batch[target_key] = []
            for target_i_dict in target:
                target_key_i = {sub_key: target_i_dict[sub_key] for sub_key in sub_keys}
                batch[target_key].append(target_key_i)

        return batch


class SelectKeyTransform:
    def __init__(self, *, key, transform, rename=None):
        self.key = key
        self.transform = transform
        self.rename = rename

    def __call__(self, target: dict):
        new_key = self.rename or self.key

        target[new_key] = self.transform(target[self.key])
        if self.rename is not None:
            del target[self.key]

        return target


class Method:
    def __init__(self, *args, name, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, obj):
        method = getattr(obj, self.name)
        ret = method(*self.args, **self.kwargs)
        return ret
