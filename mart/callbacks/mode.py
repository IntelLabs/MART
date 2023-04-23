#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from pytorch_lightning.callbacks import Callback

__all__ = ["OverrideMode"]


class OverrideMode(Callback):
    def __init__(
        self,
        training_mode: str = "train",
        validation_mode: str = "eval",
        test_mode: str = "eval",
    ):
        self.training_mode = training_mode == "train"
        self.validation_mode = validation_mode == "train"
        self.test_mode = test_mode == "train"

        self.mode = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, unused=0):
        self.mode = pl_module.training
        pl_module.train(self.training_mode)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        pl_module.train(self.mode)
        self.mode = None

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.mode = pl_module.training
        pl_module.train(self.validation_mode)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        pl_module.train(self.mode)
        self.mode = None

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.mode = pl_module.training
        pl_module.train(self.test_mode)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pl_module.train(self.mode)
        self.mode = None
