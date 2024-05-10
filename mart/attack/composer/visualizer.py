#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch
from torchvision.transforms.functional import to_pil_image


class Visualizer:
    def __call__(self, output):
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                to_pil_image(value / 255).save(f"{key}.png")
