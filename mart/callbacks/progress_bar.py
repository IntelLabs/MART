#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import tqdm
from pytorch_lightning.callbacks import TQDMProgressBar

__all__ = ["ProgressBar"]


class ProgressBar(TQDMProgressBar):
    """Display progress bar of attack iterations with the gain value."""

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.leave = False
        bar.set_description("Attack")
        bar.unit = "iter"

        return bar
