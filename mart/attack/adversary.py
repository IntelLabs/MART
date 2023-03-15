#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING, Any, Callable

import pytorch_lightning as pl
import torch

from mart.utils import silent

if TYPE_CHECKING:
    from .gradient_modifier import GradientModifier
    from .initializer import Initializer
    from .objective import Objective
    from .projector import Projector
    from .threat_model import ThreatModel

__all__ = ["Adversary", "LitPerturber"]


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
            max_epochs=1,
            limit_train_batches=kwargs.pop("max_iters", 10),
            callbacks=list(kwargs.pop("callbacks", {}).values()),
            enable_model_summary=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

        # We feed the same batch to the attack every time so we treat each step as an
        # attack iteration. As such, attackers must only run for 1 epoch and must limit
        # the number of attack steps via limit_train_batches.
        assert self.attacker.max_epochs == 1
        assert self.attacker.limit_train_batches > 0

        self.perturber = perturber or LitPerturber(**kwargs)

    @silent()
    def forward(self, **batch):
        if "model" in batch:
            # Attack for one epoch
            self.attacker.fit(model=self.perturber, train_dataloaders=cycle([batch]))

            # Enable future attacks to fit by increasing max_epochs by 1
            self.attacker.fit_loop.max_epochs += 1

        return self.perturber(**batch)


class LitPerturber(pl.LightningModule):
    """Peturbation optimization module."""

    def __init__(
        self,
        *,
        initializer: Initializer,
        optimizer: Callable,
        threat_model: ThreatModel,
        gradient_modifier: GradientModifier | None = None,
        projector: Projector | None = None,
        gain: str = "loss",
        objective: Objective | None = None,
    ):
        """_summary_

        Args:
            initializer (Initializer): To initialize the perturbation.
            optimizer (torch.optim.Optimizer): A PyTorch optimizer.
            threat_model (ThreatModel): A layer which injects perturbation to input, serving as the preprocessing layer to the target model.
            gradient_modifier (GradientModifier): To modify the gradient of perturbation.
            projector (Projector): To project the perturbation into some space.
            gain (str): Which output to use as an adversarial gain function, which is a differentiable estimate of adversarial objective. (default: loss)
            objective (Objective): A function for computing adversarial objective, which returns True or False. Optional.
        """
        super().__init__()

        self.initializer = initializer
        self.optimizer_fn = optimizer
        self.threat_model = threat_model
        self.gradient_modifier = gradient_modifier or GradientModifier()
        self.projector = projector or Projector()
        self.gain_output = gain
        self.objective_fn = objective

    def configure_optimizers(self):
        # Perturbation is lazily initialized but we need a reference to it for the optimizer
        self.perturbation = torch.nn.UninitializedBuffer(requires_grad=True)

        return self.optimizer_fn([self.perturbation])

    def training_step(self, batch, batch_idx):
        # copy batch since we modify it and it is used internally
        batch = batch.copy()
        input = batch.pop("input")
        target = batch.pop("target")
        model = batch.pop("model")

        outputs = model(input=input, target=target, **batch)
        # FIXME: This should really be just `return outputs`. Everything below here should live in the model!
        gain = outputs[self.gain_output]

        # FIXME: Make objective a part of the model...
        # objective_fn is optional, because adversaries may never reach their objective.
        if self.objective_fn is not None:
            found = self.objective_fn(**outputs)
            self.log("found", found.sum().float(), prog_bar=True)

            # No need to calculate new gradients if adversarial examples are already found.
            if len(gain.shape) > 0:
                gain = gain[~found]

        if len(gain.shape) > 0:
            gain = gain.sum()

        self.log("gain", gain, prog_bar=True)

        return gain

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # FIXME: pl.Trainer might implement some of this functionality so GradientModifier can probably go away?
        self.gradient_modifier(self.perturbation.grad)

    def forward(
        self,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        **kwargs,
    ):
        # Materialize perturbation and initialize it
        if torch.nn.parameter.is_lazy(self.perturbation):
            self.perturbation.materialize(input.shape, device=input.device, dtype=torch.float32)
            self.initializer(self.perturbation)

        # Project perturbation...
        self.projector(self.perturbation.data, input, target)

        # ...and apply threat model.
        return self.threat_model(input, target, self.perturbation)
