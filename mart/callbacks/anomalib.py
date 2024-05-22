#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import types

import torch
from lightning.pytorch.callbacks import Callback

from ..utils import MonkeyPatch, get_pylogger

logger = get_pylogger(__name__)


__all__ = ["SemanticAdversary"]


class SemanticAdversary(Callback):
    """Perturbs inputs to be adversarial under semantic contraints."""

    def __init__(
        self,
        lr: float = 5.0,
        steps: int = 100,
        restarts: int = 5,
        angle_init: float = 0,
        angle_bound: float = 90.0,
        angle_lr_mult: float = 1,
        hue_init: float = 0,
        hue_bound: float = torch.pi,
        hue_lr_mult: float = 0.02,
        sat_init: float = 0,
        sat_bound: float = 0.5,
        sat_lr_mult: float = 0.02,
    ):
        super().__init__()

        self.lr = lr
        self.steps = steps
        self.restarts = restarts

        self.angle_init = angle_init
        self.angle_bound = angle_bound
        self.angle_lr_mult = angle_lr_mult

        self.hue_init = hue_init
        self.hue_bound = hue_bound
        self.hue_lr_mult = hue_lr_mult

        self.sat_init = sat_init
        self.sat_bound = sat_bound
        self.sat_lr_mult = sat_lr_mult

    def setup(self, trainer, pl_module, stage=None):
        self._on_after_batch_transfer = pl_module.on_after_batch_transfer
        pl_module.on_after_batch_transfer = types.MethodType(
            self.on_after_batch_transfer, pl_module
        )

    def teardown(self, trainer, pl_module, stage=None):
        pl_module.on_after_batch_transfer = self._on_after_batch_transfer

    def on_after_batch_transfer(self, pl_module, batch, dataloader_idx):
        batch = self._on_after_batch_transfer(batch, dataloader_idx)

        trainer = pl_module.trainer
        if not trainer.testing:
            return batch

        # FIXME: Add semantic perturbations code here.

        return batch
