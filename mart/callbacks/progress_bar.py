#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

__all__ = ["ProgressBar"]


class ProgressBar(TQDMProgressBar):
    """Display progress bar of attack iterations with the gain value."""

    def __init__(self, disable=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if disable:
            self.disable()

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.leave = False
        bar.set_description("Attack")
        bar.unit = "iter"

        return bar

    def on_train_epoch_start(self, trainer: pl.Trainer, *_: Any) -> None:
        super().on_train_epoch_start(trainer)

        # So that it does not display negative rate.
        self.main_progress_bar.initial = 0
        # So that it does not display Epoch n.
        self.main_progress_bar.set_description("Attack")
