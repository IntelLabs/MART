#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from functools import reduce

import torch

from mart.transforms.batch_c15n import BatchC15n


class ObjectDetectionBatchConverter(BatchC15n):
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

        return input, target

    def _revert(self, input: tuple[torch.Tensor], target: tuple[dict]) -> dict:
        batch = {}

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
