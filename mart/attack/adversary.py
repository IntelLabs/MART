#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from collections import OrderedDict

import torch

from .callbacks import Callback

__all__ = ["Adversary", "NoAdversary"]


class AdversaryCallbackHookMixin(Callback):
    """Define event hooks in the Adversary Loop for callbacks."""

    callbacks = {}

    def on_run_start(self, adversary, input, target, model, **kwargs):
        """Prepare the attack loop state."""
        for _name, callback in self.callbacks.items():
            # FIXME: Skip incomplete callback instance.
            callback.on_run_start(adversary, input, target, model, **kwargs)

    def on_examine_start(self, adversary, input, target, model, **kwargs):
        for _name, callback in self.callbacks.items():
            callback.on_examine_start(adversary, input, target, model, **kwargs)

    def on_examine_end(self, adversary, input, target, model, **kwargs):
        for _name, callback in self.callbacks.items():
            callback.on_examine_end(adversary, input, target, model, **kwargs)

    def on_advance_start(self, adversary, input, target, model, **kwargs):
        for _name, callback in self.callbacks.items():
            callback.on_advance_start(adversary, input, target, model, **kwargs)

    def on_advance_end(self, adversary, input, target, model, **kwargs):
        for _name, callback in self.callbacks.items():
            callback.on_advance_end(adversary, input, target, model, **kwargs)

    def on_run_end(self, adversary, input, target, model, **kwargs):
        for _name, callback in self.callbacks.items():
            callback.on_run_end(adversary, input, target, model, **kwargs)


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
        perturber,
        optimizer,
        max_iters,
        gain,
        objective=None,
        callbacks=None,
    ):
        """_summary_

        Args:
            perturber (torch.nn.Module): _description_.
            optimizer (torch.optim.Optimizer): A PyTorch optimizer.
            max_iters (int): The max number of attack iterations.
            gain (torch.nn.Module): An adversarial gain function, which is a differentiable estimate of adversarial objective.
            objective (torch.nn.Module): A function for computing adversarial objective, which returns True or False.
            callbacks (dict): A dict of callback objects.
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

    def on_run_start(self, adversary, input, target, model, **kwargs):
        # FIXME: We should probably just register IterativeAdversary as a callback.
        # Set up the optimizer.
        self.cur_iter = 0

        # We could be at the inference/no-grad mode here.
        # Initialize lazy module
        # FIXME: Perturbers can just use on_run_start/on_run_end to initialize
        # self.perturber(input, target)
        # Initialize the Uninitialized parameters here, because self.opt needs perturber.params.
        adversary.perturber.on_run_start(adversary, input, target, model, **kwargs)

        # Split param groups by input elements, so that we can schedule optimizers individually.
        param_groups = [{"params": [param]} for param in self.perturber.parameters()]
        self.opt = self.optimizer_fn(param_groups)

        # Run callbacks.on_run_start() later, so that everything above is usable.
        #   lr_scheduler needs self.opt.
        super().on_run_start(adversary, input, target, model, **kwargs)

    def on_run_end(self, adversary, input, target, model, **kwargs):
        super().on_run_end(adversary, input, target, model, **kwargs)

        # Release optimization resources
        del self.opt

    # Disable mixed-precision optimization for attacks,
    #   since we haven't implemented it yet.
    @torch.autocast("cuda", enabled=False)
    @torch.autocast("cpu", enabled=False)
    def forward(self, input, target, model, **kwargs):
        """_summary_

        Args:
            input (_type_): _description_
            target (_type_): _description_
            model (_type_): _description_
        """

        self.on_run_start(self, input, target, model, **kwargs)

        while True:
            try:
                self.on_examine_start(self, input, target, model, **kwargs)
                self.examine(input, target, model, **kwargs)
                self.on_examine_end(self, input, target, model, **kwargs)

                # Check the done condition here, so that every update of perturbation is examined.
                if not self.done:
                    self.on_advance_start(self, input, target, model, **kwargs)
                    self.advance(input, target, model, **kwargs)
                    self.on_advance_end(self, input, target, model, **kwargs)
                    # Update cur_iter at the end so that all hooks get the correct cur_iter.
                    self.cur_iter += 1
                else:
                    break
            except StopIteration:
                break

        self.on_run_end(self, input, target, model, **kwargs)

    # Make sure we can do autograd.
    # Earlier Pytorch Lightning uses no_grad(), but later PL uses inference_mode():
    #   https://github.com/Lightning-AI/lightning/pull/12715
    @torch.enable_grad()
    @torch.inference_mode(False)
    def examine(self, input, target, model, **kwargs):
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
    def advance(self, input, target, model, **kwargs):
        """Run one attack iteration."""

        self.opt.zero_grad()

        # Do not flip the gain value, because we set maximize=True in optimizer.
        self.total_gain.backward()

        self.opt.step()


class Adversary(IterativeGenerator):
    """An adversary module which generates and applies perturbation to input."""

    def __init__(self, threat_model, *args, **kwargs):
        """_summary_

        Args:
            threat_model (torch.nn.Module): A layer which injects perturbation to input, serving as the preprocessing layer to the target model.
        """
        super().__init__(*args, **kwargs)

        self.threat_model = threat_model

    def forward(self, input, target, model=None, **kwargs):
        # Generate a perturbation only if we have a model. This will update
        # the parameters of self.perturber.
        if model is not None:
            super().forward(input, target, model, **kwargs)

        # Get perturbation and apply threat model
        perturbation = self.perturber(input, target)
        output = self.threat_model(input, target, perturbation, **kwargs)

        return output


class NoAdversary(torch.nn.Module):
    def forward(self, input, target, model=None, **kwargs):
        return input
