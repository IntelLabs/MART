#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from pytorch_lightning.callbacks import Callback

__all__ = ["ModelParamsNoGrad"]


class ModelParamsNoGrad(Callback):
    """No gradient for model parameters during attack.

    This callback should not change the result. Don't use unless an attack runs faster.
    """

    def on_train_start(self, trainer, model):
        for param in model.parameters():
            param.requires_grad_(False)

    def on_train_end(self, trainer, model):
        for param in model.parameters():
            param.requires_grad_(True)
