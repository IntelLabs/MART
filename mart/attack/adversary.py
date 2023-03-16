#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from itertools import cycle

import pytorch_lightning as pl

from mart.attack import LitPerturber
from mart.utils import silent

__all__ = ["Adversary"]


class Adversary(torch.nn.Module):
    def __init__(
        self,
        *,
        trainer: pl.Trainer | None = None,
        perturber: LitPerturber | None = None,
        **kwargs,
    ):
        """_summary_

        Args:
            trainer (Trainer): A PyTorch-Lightning Trainer object used to fit the perturber.
            perturber (LitPerturber): A LitPerturber that manages perturbations.
        """
        super().__init__()

        self.attacker = trainer or pl.Trainer(
            accelerator="auto",  # FIXME: we need to get this on the same device as input...
            num_sanity_val_steps=0,
            logger=False,
            max_epochs=0,
            limit_train_batches=kwargs.pop("max_iters", 10),
            callbacks=list(kwargs.pop("callbacks", {}).values()),
            enable_model_summary=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

        # We feed the same batch to the attack every time so we treat each step as an
        # attack iteration. As such, attackers must only run for 1 epoch and must limit
        # the number of attack steps via limit_train_batches.
        assert self.attacker.max_epochs == 0
        assert self.attacker.limit_train_batches > 0

        self.perturber = perturber or LitPerturber(**kwargs)

    @silent()
    def forward(self, **batch):
        # Adversary lives within a sequence of nn.Modules. To signal the adversary should attack, one
        # must pass a model to attack when calling the adversary.
        if "model" in batch:
            # Attack, aka fit a perturbation, for one epoch by cycling over the same input batch.
            # We use Trainer.limit_train_batches to control the number of attack iterations.
            self.attacker.fit_loop.max_epochs += 1
            self.attacker.fit(self.perturber, train_dataloaders=cycle([batch]))

        # Always use perturb the current input.
        return self.perturber(**batch)
