#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import logging
from collections import OrderedDict
from functools import partial
from itertools import cycle
from typing import TYPE_CHECKING, Any, Callable

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch.nn.modules.lazy import LazyModuleMixin

if TYPE_CHECKING:
    from .gradient_modifier import GradientModifier
    from .initializer import Initializer
    from .objective import Objective
    from .projector import Projector
    from .threat_model import ThreatModel

__all__ = ["Adversary"]


class LitPerturber(LazyModuleMixin, LightningModule):
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
        **kwargs,
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
        self.gradient_modifier = gradient_modifier
        self.projector = projector
        self.gain_output = gain
        self.objective_fn = objective

        self.perturbation = torch.nn.UninitializedParameter()

        def projector_wrapper(module, args):
            if isinstance(module.perturbation, torch.nn.UninitializedBuffer):
                raise ValueError("Perturbation must be initialized")

            input, target = args

            # FIXME: How do we get rid of .to(input.device)?
            return projector(module.perturbation.data.to(input.device), input, target)

        # Will be called before forward() is called.
        if projector is not None:
            self.register_forward_pre_hook(projector_wrapper)

    def configure_optimizers(self):
        return self.optimizer_fn([self.perturbation])

    def training_step(self, batch, batch_idx):
        # copy batch since we will modify it and it it passed around
        batch = batch.copy()
        input = batch.pop("input")
        target = batch.pop("target")
        model = batch.pop("model")

        if self.has_uninitialized_params():
            # Use this syntax because LazyModuleMixin assume non-keyword arguments
            self(input, target)

        outputs = model(input=input, target=target, **batch)
        # FIXME: This should really be just `return outputs`. Everything below here should live in the model!
        gain = outputs[self.gain_output]

        # objective_fn is optional, because adversaries may never reach their objective.
        # FIXME: Make objective a part of the model...
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

    def initialize_parameters(self, input, target):
        assert isinstance(self.perturbation, torch.nn.UninitializedParameter)

        self.perturbation.materialize(input.shape, device=input.device)

        # A backward hook that will be called when a gradient w.r.t the Tensor is computed.
        if self.gradient_modifier is not None:
            self.perturbation.register_hook(self.gradient_modifier)

        self.initializer(self.perturbation)

    def forward(self, input, target, **kwargs):
        # FIXME: Can we get rid of .to(input.device)?
        perturbation = self.perturbation.to(input.device)

        # Get perturbation and apply threat model
        # The mask projector in perturber may require information from target.
        return self.threat_model(input, target, perturbation)


class Adversary(torch.nn.Module):
    """An adversary module which generates and applies perturbation to input."""

    def __init__(
        self,
        *,
        max_iters: int = 10,
        callbacks: dict[str, Callback] | None = None,
        **perturber_kwargs,
    ):
        """_summary_

        Args:
            max_iters (int): The max number of attack iterations.
            callbacks (dict[str, Callback] | None): A dictionary of callback objects. Optional.
        """
        super().__init__()

        self.perturber_factory = partial(LitPerturber, **perturber_kwargs)

        # FIXME: how do we get a proper device?
        self.attacker_factory = partial(
            Trainer,
            accelerator="auto",
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            max_epochs=1,
            max_steps=max_iters,
            enable_model_summary=False,
            callbacks=list(callbacks.values()),  # ignore keys
            enable_checkpointing=False,
        )

    def forward(
        self,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        model: torch.nn.Module | None = None,
        **kwargs,
    ):
        # Generate a perturbation only if we have a model. This will update
        # the parameters of self.perturber
        if model is not None:
            benign_dataloader = cycle(
                [{"input": input, "target": target, "model": model, **kwargs}]
            )

            with Silence():
                self.perturber = [self.perturber_factory()]
                self.attacker_factory().fit(
                    model=self.perturber[0], train_dataloaders=benign_dataloader
                )

        # Get preturbed input (some threat models, projectors, etc. may require information from target like a mask)
        return self.perturber[0](input, target)

class Silence:
    """Suppress logging."""

    DEFAULT_NAMES = ["pytorch_lightning.utilities.rank_zero", "pytorch_lightning.accelerators.gpu"]

    def __init__(self, names=None):
        if names is None:
            names = Silence.DEFAULT_NAMES

        self.loggers = [logging.getLogger(name) for name in names]

    def __enter__(self):
        for logger in self.loggers:
            logger.propagate = False

    def __exit__(self, exc_type, exc_value, traceback):
        for logger in self.loggers:
            logger.propagate = False
