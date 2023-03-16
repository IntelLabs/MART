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
        # FIXME: It would be nice if we didn't have to create this buffer every time someone call's fit.
        self.perturbation = torch.nn.UninitializedBuffer(requires_grad=True)

        return self.optimizer_fn([self.perturbation])

    def training_step(self, batch, batch_idx):
        # copy batch since we modify it and it is used internally
        batch = batch.copy()

        # We need to evaluate the perturbation against the whole model, so call it normally to get a gain.
        model = batch.pop("model")
        outputs = model(**batch)

        # FIXME: This should really be just `return outputs`. But this might require a new sequence?
        # FIXME: Everything below here should live in the model as modules.
        gain = outputs[self.gain_output]

        # objective_fn is optional, because adversaries may never reach their objective.
        if self.objective_fn is not None:
            found = self.objective_fn(**outputs)

            # No need to calculate new gradients if adversarial examples are already found.
            if len(gain.shape) > 0:
                gain = gain[~found]

        if len(gain.shape) > 0:
            gain = gain.sum()

        return gain

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val=None, gradient_clip_algorithm=None):
        # Configuring gradient clipping in pl.Trainer is still useful, so use it.
        super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

        # FIXME: Why not loop through optimizer.param_groups?
        # FIXME: Make gradient modifier an in-place operation. Will make it easier to fix the above.
        self.perturbation.grad = self.gradient_modifier(self.perturbation.grad)

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
        # FIXME: Projector should probably be an in-place operation instead of passing .data?
        self.projector(self.perturbation.data, input, target)

        # ...and apply threat model.
        return self.threat_model(input, target, self.perturbation)
