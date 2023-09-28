#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import os

import torch
from lightning.pytorch.callbacks import Callback
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
        # Save canonical input and target for on_train_end
        self.input = batch[0]
        self.target = batch[1]

    def on_train_end(self, trainer, model):
        # FIXME: We should really just save this to outputs instead of recomputing adv_input
        with torch.no_grad():
            adv_input, _target = model(self.input, self.target)

        for img, tgt in zip(adv_input, self.target):
            fname = tgt["file_name"]
            fpath = os.path.join(self.folder, fname)
            im = self.convert(img / 255)
            im.save(fpath)
