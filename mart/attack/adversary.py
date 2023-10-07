#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from functools import partial
from itertools import cycle
from typing import TYPE_CHECKING, Any, Callable

import lightning.pytorch as pl
import torch

from mart.utils import silent

from ..optim import OptimizerFactory
from ..utils.optimization import configure_optimizers

if TYPE_CHECKING:
    from .composer import Composer
    from .enforcer import Enforcer
    from .gain import Gain
    from .gradient_modifier import GradientModifier
    from .objective import Objective

__all__ = ["Adversary"]


class Adversary(pl.LightningModule):
    """An adversary module which generates and applies perturbation to input."""

    def __init__(
        self,
        *,
        composer: Composer,
        optimizer: OptimizerFactory | Callable[[Any], torch.optim.Optimizer],
        lr_scheduler: dict | None = None,
        gain: Gain,
        gradient_modifier: GradientModifier | None = None,
        objective: Objective | None = None,
        enforcer: Enforcer | None = None,
        attacker: pl.Trainer | None = None,
        **kwargs,
    ):
        """_summary_

        Args:
            composer (Composer): A MART Composer.
            optimizer (OptimizerFactory | Callable[[Any], torch.optim.Optimizer]): A MART OptimizerFactory or partial that returns an Optimizer when given params.
            lr_scheduler (dict): A dictionary for learning rate scheduling in PyTorch Lightning.
            gain (Gain): An adversarial gain function, which is a differentiable estimate of adversarial objective.
            gradient_modifier (GradientModifier): To modify the gradient of perturbation.
            objective (Objective): A function for computing adversarial objective, which returns True or False. Optional.
            enforcer (Enforcer): A Callable that enforce constraints on the adversarial input.
            attacker (Trainer): A PyTorch-Lightning Trainer object used to fit the perturbation.
        """
        super().__init__()

        # Avoid resuming perturbation from a saved checkpoint.
        self._register_load_state_dict_pre_hook(
            # Appear to have consumed the state_dict.
            lambda state_dict, *args, **kwargs: state_dict.clear()
        )

        # Hide the composer module in a list, so that perturbation is not exported as a parameter in the model checkpoint.
        # and DDP won't try to get the uninitialized parameters of perturbation.
        self._composer = [composer]
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        if not isinstance(self.optimizer, OptimizerFactory):
            self.optimizer = OptimizerFactory(self.optimizer)
        self.gain_fn = gain
        self.gradient_modifier = gradient_modifier
        self.objective_fn = objective
        self.enforcer = enforcer

        self._attacker = attacker

        if self._attacker is None:
            # Enable attack to be late bound in forward
            self._attacker = partial(
                pl.Trainer,
                num_sanity_val_steps=0,
                logger=list(kwargs.pop("logger", {}).values()),
                max_epochs=0,
                limit_train_batches=kwargs.pop("max_iters", 10),
                callbacks=list(kwargs.pop("callbacks", {}).values()),  # dict to list of values
                enable_model_summary=False,
                enable_checkpointing=False,
                # We should disable progress bar in the progress_bar callback config if needed.
                enable_progress_bar=True,
                # detect_anomaly=True,
            )

        else:
            # We feed the same batch to the attack every time so we treat each step as an
            # attack iteration. As such, attackers must only run for 1 epoch and must limit
            # the number of attack steps via limit_train_batches.
            assert self._attacker.max_epochs == 0
            assert self._attacker.limit_train_batches > 0

    @property
    def composer(self) -> Composer:
        # Hide the composer module in a list, so that perturbation is not exported as a parameter in the model checkpoint,
        # and DDP won't try to get the uninitialized parameters of perturbation.
        return self._composer[0]

    def configure_optimizers(self):
        return configure_optimizers(self.composer, self.optimizer, self.lr_scheduler)

    def training_step(self, batch_and_model, batch_idx):
        input, target, model = batch_and_model

        # Compose adversarial examples.
        input_adv, target_adv = self.forward(input, target)

        # A model that returns output dictionary.
        outputs = model(input_adv, target_adv)

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

        # Log gain as a metric for LR scheduler to monitor, and show gain on progress bar.
        self.log("gain", gain, prog_bar=True)

        return gain

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        # Configuring gradient clipping in pl.Trainer is still useful, so use it.
        super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

        if self.gradient_modifier:
            for group in optimizer.param_groups:
                self.gradient_modifier(group["params"])

    @silent()
    def fit(self, input, target, *, model: Callable):
        # The attack also needs access to the model at every iteration.
        batch_and_model = (input, target, model)

        # Configure and reset perturbation for current inputs
        self.composer.configure_perturbation(input)

        # Attack, aka fit a perturbation, for one epoch by cycling over the same input batch.
        # We use Trainer.limit_train_batches to control the number of attack iterations.
        self.attacker.fit_loop.max_epochs += 1
        self.attacker.fit(self, train_dataloaders=cycle([batch_and_model]))

    def forward(self, input, target):
        """Compose adversarial examples and enforce the threat model."""
        input_adv = self.composer(input=input, target=target)

        if self.enforcer is not None:
            self.enforcer(input_adv, input=input, target=target)

        return input_adv, target

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
            # Lightning Fabric: `devices` selected with `CPUAccelerator` should be an int > 0
            devices = 1

        else:
            raise NotImplementedError

        self._attacker = self._attacker(accelerator=accelerator, devices=devices)

        return self._attacker

    def cpu(self):
        # PL places the LightningModule back on the CPU after fitting:
        #   https://github.com/Lightning-AI/lightning/blob/ff5361604b2fd508aa2432babed6844fbe268849/pytorch_lightning/strategies/single_device.py#L96
        #   https://github.com/Lightning-AI/lightning/blob/ff5361604b2fd508aa2432babed6844fbe268849/pytorch_lightning/strategies/ddp.py#L482
        # This is a problem when this LightningModule has parameters, so we stop this from
        # happening by ignoring the call to cpu().
        pass
