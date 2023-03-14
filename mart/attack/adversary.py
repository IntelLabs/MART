#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from collections import OrderedDict
from typing import Any
from itertools import cycle
from functools import partial

import torch
from torch.nn.modules.lazy import LazyModuleMixin
from pytorch_lightning import LightningModule, Trainer

from .callbacks import Callback
from .gain import Gain
from .objective import Objective
from .perturber import BatchPerturber, Perturber
from .threat_model import ThreatModel

__all__ = ["Adversary"]


class AdversaryCallbackHookMixin(Callback):
    """Define event hooks in the Adversary Loop for callbacks."""

    callbacks = {}

    def on_run_start(self, **kwargs) -> None:
        """Prepare the attack loop state."""
        for _name, callback in self.callbacks.items():
            # FIXME: Skip incomplete callback instance.
            # Give access of self to callbacks by `adversary=self`.
            callback.on_run_start(**kwargs)

    def on_examine_start(self, **kwargs) -> None:
        for _name, callback in self.callbacks.items():
            callback.on_examine_start(**kwargs)

    def on_examine_end(self, **kwargs) -> None:
        for _name, callback in self.callbacks.items():
            callback.on_examine_end(**kwargs)

    def on_advance_start(self, **kwargs) -> None:
        for _name, callback in self.callbacks.items():
            callback.on_advance_start(**kwargs)

    def on_advance_end(self, **kwargs) -> None:
        for _name, callback in self.callbacks.items():
            callback.on_advance_end(**kwargs)

    def on_run_end(self, **kwargs) -> None:
        for _name, callback in self.callbacks.items():
            callback.on_run_end(**kwargs)


class LitPerturber(LazyModuleMixin, LightningModule):
    """Peturbation optimization module."""

    def __init__(
        self,
        *,
        initializer: Initializer,
        optimizer: Callable,
        gradient_modifier: GradientModifier | None = None,
        projector: Projector | None = None,
        gain: str = "loss",
        **kwargs
    ):
        """_summary_

        Args:
        """
        super().__init__()

        self.initializer = initializer
        self.gradient_modifier = gradient_modifier
        self.projector = projector
        self.optimizer_fn = optimizer
        self.gain = gain

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
        input = batch.pop("input")
        target = batch.pop("target")
        model = batch.pop("model")

        if self.has_uninitialized_params():
            # Use this syntax because LazyModuleMixin assume non-keyword arguments
            self(input, target)

        outputs = model(input=input, target=target, **batch)

        return outputs[self.gain]

    def initialize_parameters(self, input, target):
        assert isinstance(self.perturbation, torch.nn.UninitializedParameter)

        self.perturbation.materialize(input.shape, device=input.device)

        # A backward hook that will be called when a gradient w.r.t the Tensor is computed.
        if self.gradient_modifier is not None:
            self.perturbation.register_hook(self.gradient_modifier)

        self.initializer(self.perturbation)


    def forward(self, input, target, **kwargs):
        # FIXME: Can we get rid of .to(input.device)?
        return self.perturbation.to(input.device)

class Adversary(torch.nn.Module):
    """An adversary module which generates and applies perturbation to input."""

    def __init__(
        self,
        *,
        threat_model: ThreatModel,
        max_iters: int = 10,
        callbacks: dict[str, Callback] | None = None,
        **perturber_kwargs,
    ):
        """_summary_

        Args:
            threat_model (ThreatModel): A layer which injects perturbation to input, serving as the preprocessing layer to the target model.
            max_iters (int): The max number of attack iterations.
            callbacks (dict[str, Callback] | None): A dictionary of callback objects. Optional.
        """
        super().__init__()

        self.threat_model = threat_model
        self.callbacks = callbacks # FIXME: Register these with trainer?
        self.perturber_factory = partial(LitPerturber, **perturber_kwargs)

        # FIXME: how do we get a proper device?
        self.attacker = Trainer(accelerator="auto", max_steps=max_iters, enable_model_summary=False)

    def forward(
        self,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        model: torch.nn.Module | None = None,
        **kwargs,
    ):
        # Generate a perturbation only if we have a model. This will update
        # the parameters of self.perturbation.
        if model is not None:
            self.perturber = [self.perturber_factory()]

            benign_dataloader = cycle([{"input": input, "target": target, "model": model, **kwargs}])
            self.attacker.fit(model=self.perturber[0], train_dataloaders=benign_dataloader)

        # Get perturbation and apply threat model
        # The mask projector in perturber may require information from target.
        perturbation = self.perturber[0](input, target)
        output = self.threat_model(input, target, perturbation)

        return output
