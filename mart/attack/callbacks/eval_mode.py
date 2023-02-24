#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from .base import Callback

__all__ = ["AttackInEvalMode"]


class AttackInEvalMode(Callback):
    """Switch the model into eval mode during attack."""

    def __init__(self):
        self.training_mode_status = None

    def on_run_start(self, *, model, **kwargs):
        self.training_mode_status = model.training
        model.train(False)

    def on_run_end(self, *, model, **kwargs):
        assert self.training_mode_status is not None

        # Resume the previous training status of the model.
        model.train(self.training_mode_status)
