#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from typing import Any

from lightning import pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.utilities.rank_zero import rank_zero_only

__all__ = ["ProgressBar"]


class ProgressBar(TQDMProgressBar):
    """Display progress bar of attack iterations with the gain value."""

    def __init__(self, *args, enable=True, **kwargs):
        if "process_position" not in kwargs:
            # Automatically place the progress bar by rank if position is not specified.
            # rank starts with 0
            rank_id = rank_zero_only.rank
            # Adversary progress bars start at position 1, because the main progress bar takes position 0.
            process_position = rank_id + 1
            kwargs["process_position"] = process_position

        super().__init__(*args, **kwargs)

        if not enable:
            self.disable()

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.leave = False
        bar.set_description("Attack")
        bar.unit = "iter"

        return bar

    def on_train_epoch_start(self, trainer: pl.Trainer, *_: Any) -> None:
        super().on_train_epoch_start(trainer)

        # So that it does not display Epoch n.
        rank_id = rank_zero_only.rank
        self.main_progress_bar.set_description(f"Attack@rank{rank_id}")
