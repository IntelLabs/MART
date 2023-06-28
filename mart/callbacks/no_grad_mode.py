#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import torch
from pytorch_lightning.callbacks import Callback

from mart import utils

logger = utils.get_pylogger(__name__)

__all__ = ["ModelParamsNoGrad"]


class ModelParamsNoGrad(Callback):
    """No gradient for model parameters during attack.

    This callback should not change the result. Don't use unless an attack runs faster.
    """

    def __init__(self, module_names: str | list[str] = None):
        if isinstance(module_names, str):
            module_names = [module_names]

        self.module_names = module_names

    def setup(self, trainer, pl_module, stage):
        if stage != "fit":
            return

        # We use setup, and not on_train_start, so that mart.optim.OptimizerFactory can ignore parameters with no gradients.
        # See: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks
        for name, param in pl_module.named_parameters():
            if any(name.startswith(module_name) for module_name in self.module_names):
                logger.info(f"Disabling gradient for {name}")
                param.requires_grad_(False)

    def teardown(self, trainer, pl_module, stage):
        for name, param in pl_module.named_parameters():
            if any(name.startswith(module_name) for module_name in self.module_names):
                # FIXME: Why is this necessary?
                param.requires_grad_(True)
