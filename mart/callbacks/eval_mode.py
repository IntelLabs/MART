#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from pytorch_lightning.callbacks import Callback

from mart import utils

logger = utils.get_pylogger(__name__)

__all__ = ["AttackInEvalMode"]


class AttackInEvalMode(Callback):
    """Switch the model into eval mode during attack."""

    def __init__(self, module_classes: type | list[type]):
        # FIXME: convert strings to classes using hydra.utils.get_class? This will clean up some verbosity in configuration but will require importing hydra in this callback.
        if isinstance(module_classes, type):
            module_classes = [module_classes]

        self.module_classes = tuple(module_classes)

    def setup(self, trainer, pl_module, stage):
        # Log to the console so the user can see visually see which modules will be in eval mode during training.
        for name, module in pl_module.named_modules():
            if isinstance(module, self.module_classes):
                logger.info(
                    f"Setting eval mode for {name} ({module.__class__.__module__}.{module.__class__.__name__})"
                )

    def on_train_epoch_start(self, trainer, pl_module):
        # We must use on_train_epoch_start because PL will set pl_module to train mode right before this callback.
        # See: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks
        for name, module in pl_module.named_modules():
            if isinstance(module, self.module_classes):
                module.eval()

    def on_train_epoch_end(self, trainer, pl_module):
        # FIXME: Why is this necessary?
        for name, module in pl_module.named_modules():
            if isinstance(module, self.module_classes):
                module.train()
