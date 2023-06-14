#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import types
from collections import OrderedDict

import torch
from hydra.utils import instantiate

__all__ = ["Module"]


class Module(torch.nn.Module):
    """A magic Module that can override forward."""

    def __new__(cls, *args, _path_: str, **kwargs):
        # TODO: Add support for _load_state_dict_
        # TODO: Add support for _freeze_

        cfg = {"_target_": _path_}
        module = instantiate(cfg, *args, **kwargs)

        module._forward = module.forward
        module.forward = types.MethodType(Module.forward, module)

        return module

    @staticmethod
    def forward(
        self,
        *args,
        train_mode: bool = True,
        inference_mode: bool = False,
        **kwargs,
    ):
        old_train_mode = self.training

        # FIXME: Would be nice if this was a context...
        self.train(train_mode)
        with torch.inference_mode(mode=inference_mode):
            ret = self._forward(*args, **kwargs)
        self.train(old_train_mode)

        return ret
