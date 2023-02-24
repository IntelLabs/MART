#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from .base import Callback

__all__ = ["ModelParamsNoGrad"]


class ModelParamsNoGrad(Callback):
    """No gradient for model parameters during attack.

    This callback should not change the result. Don't use unless an attack runs faster.
    """

    def on_run_start(self, model=None, **kwargs):
        for param in model.parameters():
            param.requires_grad_(False)

    def on_run_end(self, model=None, **kwargs):
        for param in model.parameters():
            param.requires_grad_(True)
