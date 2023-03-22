#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from .callbacks import Callback
from .gain import Gain
from .objective import Objective
from .perturber import BatchPerturber, Perturber
from .threat_model import Composer, Enforcer

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


class IterativeGenerator(AdversaryCallbackHookMixin, torch.nn.Module):
    """The attack optimization loop.

    This class implements the following loop structure:

    .. code-block:: python

        on_run_start()

        while true:
            on_examine_start()
            examine()
            on_examine_end()

            if not done:
                on_advance_start()
                advance()
                on_advance_end()
            else:
                break

        on_run_end()
    """

    def __init__(
        self,
        *,
        perturber: BatchPerturber | Perturber,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        gain: Gain,
        objective: Objective | None = None,
        callbacks: dict[str, Callback] | None = None,
    ):
        """_summary_

        Args:
            perturber (BatchPerturber | Perturber): A module that stores perturbations.
            optimizer (torch.optim.Optimizer): A PyTorch optimizer.
            max_iters (int): The max number of attack iterations.
            gain (Gain): An adversarial gain function, which is a differentiable estimate of adversarial objective.
            objective (Objective | None): A function for computing adversarial objective, which returns True or False. Optional.
            callbacks (dict[str, Callback] | None): A dictionary of callback objects. Optional.
        """
        super().__init__()

        self.perturber = perturber
        self.optimizer_fn = optimizer

        self.max_iters = max_iters
        self.callbacks = OrderedDict()

        # Register perturber as callback if it implements Callback interface
        if isinstance(self.perturber, Callback):
            # FIXME: Use self.perturber.__class__.__name__ as key?
            self.callbacks["_perturber"] = self.perturber

        if callbacks is not None:
            self.callbacks.update(callbacks)

        self.objective_fn = objective
        # self.gain is a tensor.
        self.gain_fn = gain

    @property
    def done(self) -> bool:
        # Reach the max iteration;
        if self.cur_iter >= self.max_iters:
            return True

        # All adv. examples are found;
        if hasattr(self, "found") and bool(self.found.all()) is True:
            return True

        # Compatible with models which return None gain when objective is reached.
        # TODO: Remove gain==None stopping criteria in all models,
        #       because the BestPerturbation callback relies on gain to determine which pert is the best.
        if self.gain is None:
            return True

        return False

    def on_run_start(
        self,
        *,
        adversary: torch.nn.Module,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        model: torch.nn.Module,
        **kwargs,
    ):
        super().on_run_start(
            adversary=adversary, input=input, target=target, model=model, **kwargs
        )

        # FIXME: We should probably just register IterativeAdversary as a callback.
        # Set up the optimizer.
        self.cur_iter = 0

        # param_groups with learning rate and other optim params.
        param_groups = self.perturber.parameter_groups()

        self.opt = self.optimizer_fn(param_groups)

    def on_run_end(
        self,
        *,
        adversary: torch.nn.Module,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        model: torch.nn.Module,
        **kwargs,
    ):
        super().on_run_end(adversary=adversary, input=input, target=target, model=model, **kwargs)

        # Release optimization resources
        del self.opt

    # Disable mixed-precision optimization for attacks,
    #   since we haven't implemented it yet.
    @torch.autocast("cuda", enabled=False)
    @torch.autocast("cpu", enabled=False)
    def forward(
        self,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        model: torch.nn.Module,
        **kwargs,
    ):

        self.on_run_start(adversary=self, input=input, target=target, model=model, **kwargs)

        while True:
            try:
                self.on_examine_start(
                    adversary=self, input=input, target=target, model=model, **kwargs
                )
                self.examine(input=input, target=target, model=model, **kwargs)
                self.on_examine_end(
                    adversary=self, input=input, target=target, model=model, **kwargs
                )

                # Check the done condition here, so that every update of perturbation is examined.
                if not self.done:
                    self.on_advance_start(
                        adversary=self,
                        input=input,
                        target=target,
                        model=model,
                        **kwargs,
                    )
                    self.advance(
                        input=input,
                        target=target,
                        model=model,
                        **kwargs,
                    )
                    self.on_advance_end(
                        adversary=self,
                        input=input,
                        target=target,
                        model=model,
                        **kwargs,
                    )
                    # Update cur_iter at the end so that all hooks get the correct cur_iter.
                    self.cur_iter += 1
                else:
                    break
            except StopIteration:
                break

        self.on_run_end(adversary=self, input=input, target=target, model=model, **kwargs)

    # Make sure we can do autograd.
    # Earlier Pytorch Lightning uses no_grad(), but later PL uses inference_mode():
    #   https://github.com/Lightning-AI/lightning/pull/12715
    @torch.enable_grad()
    @torch.inference_mode(False)
    def examine(
        self,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        model: torch.nn.Module,
        **kwargs,
    ):
        """Examine current perturbation, update self.gain and self.found."""

        # Clone tensors for autograd, in case it was created in the inference mode.
        # FIXME: object detection uses non-pure-tensor data, but it may have cloned somewhere else implicitly?
        if isinstance(input, torch.Tensor):
            input = input.clone()
        if isinstance(target, torch.Tensor):
            target = target.clone()

        # Set model as None, because no need to update perturbation.
        # Save everything to self.outputs so that callbacks have access to them.
        self.outputs = model(input=input, target=target, model=None, **kwargs)

        # Use CallWith to dispatch **outputs.
        self.gain = self.gain_fn(**self.outputs)

        # objective_fn is optional, because adversaries may never reach their objective.
        if self.objective_fn is not None:
            self.found = self.objective_fn(**self.outputs)
            if self.gain.shape == torch.Size([]):
                # A reduced gain value, not an input-wise gain vector.
                self.total_gain = self.gain
            else:
                # No need to calculate new gradients if adversarial examples are already found.
                self.total_gain = self.gain[~self.found].sum()
        else:
            self.total_gain = self.gain.sum()

    # Make sure we can do autograd.
    @torch.enable_grad()
    @torch.inference_mode(False)
    def advance(
        self,
        *,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        model: torch.nn.Module,
        **kwargs,
    ):
        """Run one attack iteration."""

        self.opt.zero_grad()

        # Do not flip the gain value, because we set maximize=True in optimizer.
        self.total_gain.backward()

        self.opt.step()


class Adversary(IterativeGenerator):
    """An adversary module which generates and applies perturbation to input."""

    def __init__(self, *, composer: Composer, enforcer: Enforcer, **kwargs):
        """_summary_

        Args:
            composer (Composer): A module which composes adversarial examples from input and perturbation.
            enforcer (Enforcer): A module which checks if adversarial examples satisfy constraints.
        """
        super().__init__(**kwargs)

        self.composer = composer
        self.enforcer = enforcer

    def forward(
        self,
        input: torch.Tensor | tuple,
        target: torch.Tensor | dict[str, Any] | tuple,
        model: torch.nn.Module | None = None,
        **kwargs,
    ):
        # Generate a perturbation only if we have a model. This will update
        # the parameters of self.perturber.
        if model is not None:
            super().forward(input=input, target=target, model=model, **kwargs)

        # Get perturbation and apply threat model
        # The mask projector in perturber may require information from target.
        perturbation = self.perturber(input, target)
        output = self.composer(input, target, perturbation)

        if model is not None:
            # We only enforce constraints after the attack optimization ends.
            self.enforcer(output, input, target)

        return output
