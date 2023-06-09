#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from pytorch_lightning.callbacks import Callback

from mart import utils

logger = utils.get_pylogger(__name__)

__all__ = ["AttackInEvalMode"]


class AttackInEvalMode(Callback):
    """Switch the model into eval mode during attack."""

    def __init__(self, *module_kinds):
        self.module_kinds = module_kinds

    def setup(self, trainer, pl_module, stage):
        # Log to the console so the user can see visually see which modules will be in eval mode during training.
        for name, module in pl_module.named_modules():
            module_kind = module.__class__.__name__
            if module_kind in self.module_kinds:
                logger.info(f"Setting eval mode for {name} ({module_kind})")

    def on_train_epoch_start(self, trainer, pl_module):
        # We must use on_train_epoch_start because PL will set pl_module to train mode right before this callback.
        for name, module in pl_module.named_modules():
            module_kind = module.__class__.__name__
            if module_kind in self.module_kinds:
                module.eval()

    def on_train_epoch_end(self, trainer, pl_module):
        # FIXME: Why is this necessary?
        for name, module in pl_module.named_modules():
            module_kind = module.__class__.__name__
            if module_kind in self.module_kinds:
                module.train()
