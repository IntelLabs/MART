#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from pytorch_lightning.callbacks import Callback

__all__ = ["AttackInEvalMode"]


class AttackInEvalMode(Callback):
    """Switch the model into eval mode during attack."""

    def __init__(self):
        self.training_mode_status = None

    def on_train_start(self, trainer, model):
        self.training_mode_status = model.training
        model.train(False)

    def on_train_end(self, trainer, model):
        assert self.training_mode_status is not None

        # Resume the previous training status of the model.
        model.train(self.training_mode_status)
