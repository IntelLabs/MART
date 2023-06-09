#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from operator import attrgetter

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from mart import utils

logger = utils.get_pylogger(__name__)

__all__ = ["FreezeModule"]


class FreezeModule(Callback):
    def __init__(
        self,
        module="backbone",
    ):
        self.name = module

    def setup(self, trainer, pl_module, stage):
        module = attrgetter(self.name)(pl_module.model)

        if not isinstance(module, torch.nn.Module):
            raise MisconfigurationException(
                f"The LightningModule should have a nn.Module `{self.name}` attribute"
            )

        for name, param in module.named_parameters():
            logger.debug(f"Disabling gradient for {name}")
            param.requires_grad_(False)

        for name, module in module.named_modules():
            module_kind = module.__class__.__name__
            if "BatchNorm" in module_kind:
                logger.info(f"Setting eval mode for {name} ({module_kind})")

    def on_train_epoch_start(self, trainer, pl_module):
        module = attrgetter(self.name)(pl_module.model)

        if not isinstance(module, torch.nn.Module):
            raise MisconfigurationException(
                f"The LightningModule should have a nn.Module `{self.name}` attribute"
            )

        for name, module in module.named_modules():
            module_kind = module.__class__.__name__
            if "BatchNorm" in module_kind or "Dropout" in module_kind:
                module.eval()
