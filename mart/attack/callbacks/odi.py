#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import torch

from .base import Callback

__all__ = ["OutputDiversifiedInitialization"]


class OutputDiversifiedInitialization(Callback):
    """_summary_
    Args:
        Callback (_type_): _description_
    """

    def __init__(self, nb_iters, gain_fn, learning_rate) -> None:
        self.nb_iters = nb_iters
        self.gain_fn = gain_fn
        self.learning_rate = learning_rate

    def on_run_start(self, adversary, input, target, model, **kwargs):
        self.original_gain_fn = adversary.gain_fn
        # TODO: Fetch the original learning rate.
        self.original_learning_rates = [pg["lr"] for pg in adversary.opt.param_groups]

    def on_advance_start(self, adversary, input, target, model, **kwargs):
        """_summary_
        1. Recalculate adversary.gain with self.gain_fn() using adversary.outputs, for backward();
        2. Set learning rate as self.learning rate;
        Otherwise, restore the original learning rate.
        """
        if adversary.cur_iter == 0:
            # Set the learning rate.
            for pg in adversary.opt.param_groups:
                pg["lr"] = self.learning_rate

        elif adversary.cur_iter < self.nb_iters:
            # Recalculate gain
            # Make sure we can do autograd.
            with torch.enable_grad(), torch.inference_mode(mode=False):
                adversary.gain = self.gain_fn(**adversary.outputs)

        elif adversary.cur_iter == self.nb_iters:
            # Restore the original learning rate.
            for i, lr in enumerate(self.original_learning_rates):
                adversary.opt.param_groups[i]["lr"] = lr
