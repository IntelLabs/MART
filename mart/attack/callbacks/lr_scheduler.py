#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import torch
from torch.optim import Optimizer

from .base import Callback

__all__ = ["LearningRateSchedulerSingle", "LearningRateScheduler"]


class LearningRateSchedulerSingle(Callback):
    """The scheduler will be called at the end of each attack iteration.

    For an epoch-level scheduler defined in PyTorch (e.g.
    torch.optim.lr_scheduler.ReduceLROnPlateau), we should regard epoch as attack_iteration here.
    This scheduler adjusts learning rate for all inputs, working with adversary.gain_fn() which
    returns reduced gain, rather than input-wise gain.
    """

    def __init__(self, scheduler, frequency=1, monitor_gain=False, skip_iters=0):
        """_summary_
        Args:
            scheduler (object): Any learning rate scheduler in PyTorch.
            frequency (int, optional): Run scheduler.step() at every [frequency] iterations. Defaults to 1.
            monitor_gain (bool, optional): Run scheduler.step(gain) if true, else run scheduler.step(). Defaults to False.
            skip_iters (int, optional): Skip several iterations in the beginning for something like Output Diversified Initialization. Defaults to 0.
        """
        self.scheduler_fn = scheduler
        self.frequency = frequency
        self.monitor_gain = monitor_gain
        self.skip_iters = skip_iters

    def on_run_start(self, adversary, input, target, model, **kwargs):
        self.scheduler = self.scheduler_fn(adversary.opt)

    def on_advance_end(self, adversary, input, target, model, **kwargs):
        # We may skip some iterations in the beginning; cur_iters starts at zero.
        if adversary.cur_iter < self.skip_iters:
            return
        if adversary.cur_iter % self.frequency != 0:
            return

        if self.monitor_gain:
            self.scheduler.step(float(adversary.gain))
        else:
            self.scheduler.step()


class LearningRateScheduler(Callback):
    """The scheduler will be called at the end of each attack iteration.

    For an epoch-level scheduler defined in PyTorch (e.g.
    torch.optim.lr_scheduler.ReduceLROnPlateau), we should regard epoch as attack_iteration here.
    """

    def __init__(self, scheduler, frequency=1, monitor_gain=False, skip_iters=0):
        """_summary_
        Args:
            scheduler (object): Any learning rate scheduler in PyTorch.
            frequency (int, optional): Run scheduler.step() at every [frequency] iterations. Defaults to 1.
            monitor_gain (bool, optional): Run scheduler.step(gain) if true, else run scheduler.step(). Defaults to False.
            skip_iters (int, optional): Skip several iterations in the beginning for something like Output Diversified Initialization. Defaults to 0.
        """
        self.scheduler_fn = scheduler
        self.frequency = frequency
        self.monitor_gain = monitor_gain
        self.skip_iters = skip_iters

    def on_run_start(self, adversary, input, target, model, **kwargs):
        """Create individual schedulers for each input.

        A pseudo_optimizer for the LR Scheduler must satisfy two requirements:
           1. isinstance(optimizer, Optimizer)
           2. optimizer.param_groups is there.
        """

        self.schedulers = []
        for param_group in adversary.opt.param_groups:
            pseudo_optimizer = Optimizer([param_group], {})
            scheduler = self.scheduler_fn(pseudo_optimizer)
            self.schedulers.append(scheduler)

    def on_advance_end(self, adversary, input, target, model, **kwargs):
        # We may skip some iterations in the beginning; cur_iters starts at zero.
        if adversary.cur_iter < self.skip_iters:
            return

        # Don't update LR if an adversarial example is found.
        idx_to_step = torch.where(~adversary.found)[0]
        for i in idx_to_step:
            scheduler = self.schedulers[i]
            gain = adversary.gain[i]

            if adversary.cur_iter % self.frequency != 0:
                continue

            if self.monitor_gain:
                scheduler.step(float(gain))
            else:
                scheduler.step()

    def on_run_end(self, adversary, input, target, model, **kwargs):
        lr_list = [adversary.opt.param_groups[i]["lr"] for i in range(len(input))]
        # TODO: Log the final LR if necessary.
