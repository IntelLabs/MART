#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException

__all__ = ["FreezeModule"]


class FreezeModule(Callback):
    def __init__(
        self,
        module="backbone",
    ):
        self.name = module

    def setup(self, trainer, pl_module, stage):
        module = getattr(pl_module.model, self.name, None)

        if module is None or not isinstance(module, torch.nn.Module):
            raise MisconfigurationException(
                f"The LightningModule should have a nn.Module `{self.name}` attribute"
            )

        for param in module.parameters():
            param.requires_grad_(False)

    def on_train_epoch_start(self, trainer, pl_module):
        module = getattr(pl_module.model, self.name, None)

        if module is None or not isinstance(module, torch.nn.Module):
            raise MisconfigurationException(
                f"The LightningModule should have a nn.Module `{self.name}` attribute"
            )

        module.eval()
