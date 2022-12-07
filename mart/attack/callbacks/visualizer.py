#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import os

import torch
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

    def on_run_end(self, adversary, input, target, model, **kwargs):
        perturbation = adversary.perturber(input, target)
        adv_input = adversary.threat_model(input, target, perturbation, **kwargs)

        for img, tgt in zip(adv_input, target):
            fname = tgt["file_name"]
            fpath = os.path.join(self.folder, fname)
            im = self.convert(img / 255)
            im.save(fpath)
