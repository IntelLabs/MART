#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import os

from pytorch_lightning.callbacks import Callback
from torchvision.transforms import ToPILImage

__all__ = ["PerturbedImageVisualizer"]


class PerturbedImageVisualizer(Callback):
    """Save adversarial images as files."""

    def __init__(self, folder):
        super().__init__()

        # FIXME: This should use the Trainer's logging directory.
        self.folder = folder
        self.convert = ToPILImage()

        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        # Save input and target for on_train_end
        self.input = batch["input"]
        self.target = batch["target"]

    def on_train_end(self, trainer, model):
        # FIXME: We should really just save this to outputs instead of recomputing adv_input
        adv_input = model(self.input, self.target)

        for img, tgt in zip(adv_input, self.target):
            fname = tgt["file_name"]
            fpath = os.path.join(self.folder, fname)
            im = self.convert(img / 255)
            im.save(fpath)
