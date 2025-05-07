#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import logging
from typing import Sequence

from lightning.pytorch.callbacks import Callback

from ..nn.nn import DotDict

logger = logging.getLogger(__name__)

__all__ = ["Logging"]


class Logging(Callback):
    """For models returning a dictionary, we can configure the callback to log scalars from the
    outputs, calculate and log metrics."""

    def __init__(
        self,
        train_step_log: Sequence | dict = None,
        val_step_log: Sequence | dict = None,
        test_step_log: Sequence | dict = None,
    ):
        super().__init__()

        # Be backwards compatible by turning list into dict where each item is its own key-value
        if isinstance(train_step_log, (list, tuple)):
            train_step_log = {item: {"key": item, "prog_bar": True} for item in train_step_log}
        train_step_log = train_step_log or {}

        # Be backwards compatible by turning list into dict where each item is its own key-value
        if isinstance(val_step_log, (list, tuple)):
            val_step_log = {item: {"key": item, "prog_bar": True} for item in val_step_log}
        val_step_log = val_step_log or {}

        # Be backwards compatible by turning list into dict where each item is its own key-value
        if isinstance(test_step_log, (list, tuple)):
            test_step_log = {item: {"key": item, "prog_bar": True} for item in test_step_log}
        test_step_log = test_step_log or {}

        self.step_log = {
            "train": train_step_log,
            "val": val_step_log,
            "test": test_step_log,
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        return self.on_batch_end(outputs, prefix="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        return self.on_batch_end(outputs, prefix="val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        return self.on_batch_end(outputs, prefix="test")

    #
    # Utilities
    #
    def on_batch_end(self, outputs, *, prefix: str):
        # Convert to DotDict, so that we can use a dot-connected string as a key to find a value deep in the dictionary.
        outputs = DotDict(outputs)

        step_log = self.step_log[prefix]
        for log_name, cfg in step_log.items():
            key, prog_bar = cfg["key"], cfg["prog_bar"]
            self.log(f"{prefix}/{log_name}", outputs[key], prog_bar=prog_bar)
