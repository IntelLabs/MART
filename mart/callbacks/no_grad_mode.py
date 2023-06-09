#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from operator import attrgetter

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from mart import utils

logger = utils.get_pylogger(__name__)

__all__ = ["ModelParamsNoGrad"]


class ModelParamsNoGrad(Callback):
    """No gradient for model parameters during attack.

    This callback should not change the result. Don't use unless an attack runs faster.
    """

    def __init__(self, pl_module_attr: str = None):
        self._attr = pl_module_attr

    def get_module(self, pl_module):
        module = pl_module
        if self._attr is not None:
            module = attrgetter(self._attr)(module)

        if not isinstance(module, torch.nn.Module):
            raise MisconfigurationException(
                f"The LightningModule should have a nn.Module `{self._attr}` attribute"
            )

        return module

    def setup(self, trainer, pl_module, stage):
        module = self.get_module(pl_module)

        for name, param in module.named_parameters():
            logger.debug(f"Disabling gradient for {name}")
            param.requires_grad_(False)

    def teardown(self, trainer, pl_module, stage):
        module = self.get_module(pl_module)

        for param in module.parameters():
            param.requires_grad_(True)
