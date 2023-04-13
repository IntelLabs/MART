#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from functools import partial
from itertools import cycle
from typing import TYPE_CHECKING, Any, Callable

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from mart.utils import silent

from .gradient_modifier import GradientModifier
from .projector import Projector

if TYPE_CHECKING:
    from .composer import Composer
    from .enforcer import Enforcer
    from .gain import Gain
    from .initializer import Initializer
    from .objective import Objective

__all__ = ["Adversary", "UniversalAdversary"]


class Adversary(pl.LightningModule):
    """An adversary module which generates and applies perturbation to input."""

    def __init__(
        self,
        *,
        initializer: Initializer,
        optimizer: Callable,
        composer: Composer,
        gain: Gain,
        gradient_modifier: GradientModifier | None = None,
        projector: Projector | None = None,
        objective: Objective | None = None,
        enforcer: Enforcer | None = None,
        attacker: pl.Trainer | None = None,
        **kwargs,
    ):
        """_summary_

        Args:
            initializer (Initializer): To initialize the perturbation.
            optimizer (torch.optim.Optimizer): A PyTorch optimizer.
            composer (Composer): A module which composes adversarial input from input and perturbation.
            gain (Gain): An adversarial gain function, which is a differentiable estimate of adversarial objective.
            gradient_modifier (GradientModifier): To modify the gradient of perturbation.
            projector (Projector): To project the perturbation into some space.
            objective (Objective): A function for computing adversarial objective, which returns True or False. Optional.
            enforcer (Enforcer): A Callable that enforce constraints on the adversarial input.
            attacker (Trainer): A PyTorch-Lightning Trainer object used to fit the perturbation.
        """
        super().__init__()

        self.initializer = initializer
        self.optimizer_fn = optimizer
        self.composer = composer
        self.gradient_modifier = gradient_modifier or GradientModifier()
        self.projector = projector or Projector()
        self.gain_fn = gain
        self.objective_fn = objective
        self.enforcer = enforcer

        self._attacker = attacker

        if self._attacker is None:
            # Enable attack to be late bound in forward
            self._attacker = partial(
                pl.Trainer,
                num_sanity_val_steps=0,
                logger=False,
                max_epochs=0,
                limit_train_batches=kwargs.pop("max_iters", 10),
                callbacks=list(kwargs.pop("callbacks", {}).values()),  # dict to list of values
                enable_model_summary=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

        else:
            # We feed the same batch to the attack every time so we treat each step as an
            # attack iteration. As such, attackers must only run for 1 epoch and must limit
            # the number of attack steps via limit_train_batches.
            assert self._attacker.max_epochs == 0
            assert self._attacker.limit_train_batches > 0

        self.perturbation = None

    def configure_perturbation(self, input: torch.Tensor | tuple):
        def create_and_initialize(inp):
            pert = torch.empty_like(inp, dtype=torch.float, requires_grad=True)
            self.initializer(pert)
            return pert

        if isinstance(input, tuple):
            self.perturbation = tuple(create_and_initialize(inp) for inp in input)
        elif isinstance(input, torch.Tensor):
            self.perturbation = create_and_initialize(input)
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        if self.perturbation is None:
            raise MisconfigurationException("You need to call configure_perturbation before fit.")

        params = self.perturbation
        if not isinstance(params, tuple):
            # FIXME: Should we treat the batch dimension as independent parameters?
            params = (params,)

        return self.optimizer_fn(params)

    def training_step(self, batch, batch_idx):
        # copy batch since we modify it and it is used internally
        batch = batch.copy()

        # We need to evaluate the perturbation against the whole model, so call it normally to get a gain.
        model = batch.pop("model")
        outputs = model(**batch)

        # FIXME: This should really be just `return outputs`. But this might require a new sequence?
        # FIXME: Everything below here should live in the model as modules.
        # Use CallWith to dispatch **outputs.
        gain = self.gain_fn(**outputs)

        # objective_fn is optional, because adversaries may never reach their objective.
        if self.objective_fn is not None:
            found = self.objective_fn(**outputs)

            # No need to calculate new gradients if adversarial examples are already found.
            if len(gain.shape) > 0:
                gain = gain[~found]

        if len(gain.shape) > 0:
            gain = gain.sum()

        return gain

    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        # Configuring gradient clipping in pl.Trainer is still useful, so use it.
        super().configure_gradient_clipping(
            optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
        )

        for group in optimizer.param_groups:
            self.gradient_modifier(group["params"])

    @silent()
    def forward(self, **batch):
        # Adversary lives within a sequence of model. To signal the adversary should attack, one
        # must pass a model to attack when calling the adversary. Since we do not know where the
        # Adversary lives inside the model, we also need the remaining sequence to be able to
        # get a loss.
        if "model" in batch and "sequence" in batch:
            self._attack(**batch)

        # Always use perturb the current input.
        if self.perturbation is None:
            raise MisconfigurationException(
                "You need to call the configure_perturbation before forward."
            )

        self.projector(self.perturbation, **batch)
        input_adv = self.composer(self.perturbation, **batch)

        # Enforce constraints after the attack optimization ends.
        if "model" in batch and "sequence" in batch:
            self.enforcer(input_adv, **batch)

        return input_adv

    def _attack(self, input, **batch):
        # Configure and reset perturbation for inputs
        self.configure_perturbation(input)

        # Attack, aka fit a perturbation, for one epoch by cycling over the same input batch.
        # We use Trainer.limit_train_batches to control the number of attack iterations.
        batch = {"input": input, **batch}
        self.attacker.fit_loop.max_epochs += 1
        self.attacker.fit(self, train_dataloaders=cycle([batch]))

    @property
    def attacker(self):
        if not isinstance(self._attacker, partial):
            return self._attacker

        # Convert torch.device to PL accelerator
        if self.device.type == "cuda":
            accelerator = "gpu"
            devices = [self.device.index]
        elif self.device.type == "cpu":
            accelerator = "cpu"
            devices = None
        else:
            raise NotImplementedError

        self._attacker = self._attacker(accelerator=accelerator, devices=devices)

        return self._attacker


class UniversalAdversary(Adversary):
    def __init__(self, *args, size: tuple, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a dummy parameter so that optimizers work
        self.input = torch.nn.Parameter(torch.empty(size))

    def configure_perturbation(self, input: torch.Tensor | tuple):
        # Universal perturbations can only be initialized once.
        if self.perturbation is not None:
            return

        super().configure_perturbation(self.input)

    def _attack(self, *, input, step, **batch):
        # Configure and reset perturbation for inputs
        self.configure_perturbation(input)

        # Only attack in training
        if step != "training":
            return

        super()._attack(step=step, input=input, **batch)
