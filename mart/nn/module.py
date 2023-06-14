#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch
from hydra.utils import instantiate

__all__ = ["Module"]


class Module(torch.nn.Module):
    def __init__(self, _path_: str, *args, **kwargs):
        super().__init__()

        # TODO: Add _load_state_dict_
        # TODO: Add _freeze_

        cfg = {"_target_": _path_}
        self.module = instantiate(cfg, *args, **kwargs)

    def forward(
        self,
        *args,
        train_mode: bool = True,
        inference_mode: bool = False,
        **kwargs,
    ):
        old_train_mode = self.module.training

        # FIXME: Would be nice if this was a context...
        self.module.train(train_mode)
        with torch.inference_mode(mode=inference_mode):
            ret = self.module(*args, **kwargs)
        self.module.train(old_train_mode)

        return ret
