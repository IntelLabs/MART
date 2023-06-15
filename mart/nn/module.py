#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import types
from collections import OrderedDict
from contextlib import nullcontext

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

        module._forward_ = module.forward
        module.forward = types.MethodType(Module.forward, module)

        return module

    @staticmethod
    def forward(
        self,
        *args,
        _train_mode_: bool | None = None,
        _inference_mode_: bool | None = None,
        **kwargs,
    ):
        old_train_mode = self.training

        if _train_mode_ is not None:
            self.train(_train_mode_)

        inference_mode = nullcontext()
        if _inference_mode_ is not None:
            inference_mode = torch.inference_mode(mode=_inference_mode_)

        with inference_mode:
            ret = self._forward_(*args, **kwargs)

        if _train_mode_ is not None:
            self.train(old_train_mode)

        return ret
