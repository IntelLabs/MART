#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_only

__all__ = ["ProgressBar"]


class ProgressBar(TQDMProgressBar):
    """Display progress bar of attack iterations with the gain value."""

    def __init__(self, disable=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if disable:
            self.disable()

        # rank starts with 0
        rank_id = rank_zero_only.rank
        # Adversary progress bars start at position 1, because the main progress bar takes position 0.
        self._process_position = rank_id + 1

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
        rank_id = rank_zero_only.rank
        self.main_progress_bar.set_description(f"Attack@rank{rank_id}")

    def get_metrics(self, *args, **kwargs):
        metrics = super().get_metrics(*args, **kwargs)
        # Display gain value in progress bar.
        metrics["gain"] = metrics.pop("loss")
        return metrics
