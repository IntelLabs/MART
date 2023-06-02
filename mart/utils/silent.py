#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
from contextlib import ContextDecorator

__all__ = ["silent"]


class silent(ContextDecorator):
    """Suppress logging."""

    DEFAULT_NAMES = ["pytorch_lightning.utilities.rank_zero", "pytorch_lightning.accelerators.gpu"]

    def __init__(self, names=None):
        if names is None:
            names = silent.DEFAULT_NAMES

        self.loggers = [logging.getLogger(name) for name in names]

    def __enter__(self):
        for logger in self.loggers:
            logger.propagate = False

    def __exit__(self, exc_type, exc_value, traceback):
        for logger in self.loggers:
            logger.propagate = False
