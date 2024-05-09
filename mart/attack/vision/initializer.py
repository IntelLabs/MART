#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch
import torchvision
import torchvision.transforms.functional as F

from ...utils import pylogger
from ..initializer import Initializer

logger = pylogger.get_pylogger(__name__)


class Image(Initializer):
    def __init__(self, path: str, scale: int = 1):
        self.image = torchvision.io.read_image(path, torchvision.io.ImageReadMode.RGB) / scale

    @torch.no_grad()
    def initialize_(self, parameter: torch.Tensor) -> None:
        image = self.image

        if image.shape != parameter.shape:
            logger.info(f"Resizing image from {image.shape} to {parameter.shape}...")
            image = F.resize(image, parameter.shape[1:])

        parameter.copy_(image)
