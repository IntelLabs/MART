#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import os

from torchvision.transforms import ToPILImage

from .base import Callback

__all__ = ["PerturbedImageVisualizer"]


class PerturbedImageVisualizer(Callback):
    """Save adversarial images as files."""

    def __init__(self, folder):
        super().__init__()

        self.folder = folder
        self.convert = ToPILImage()

        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

    def on_run_end(self, *, adversary, input, target, model, **kwargs):
        adv_input = adversary(input=input, target=target, model=None, **kwargs)

        for img, tgt in zip(adv_input, target):
            fname = tgt["file_name"]
            fpath = os.path.join(self.folder, fname)
            im = self.convert(img / 255)
            im.save(fpath)
