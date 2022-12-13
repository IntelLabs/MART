#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import torch

from .base import Callback

__all__ = ["BestPerturbation"]


class BestPerturbation(Callback):
    """Save the best perturbation so far by monitoring the gain value."""

    def __init__(self):
        pass

    def teardown(self):
        del self.best_perturbation
        del self.max_gain

    def reset(self, adversary, input, target, model, **kwargs):
        # Initialize max_gain and best_pert.
        self.max_gain = torch.tensor([-torch.inf] * len(input), device=input[0].device)
        self.best_perturbation = [None] * len(input)

    def on_run_start(self, adversary, input, target, model, **kwargs):
        self.reset(adversary, input, target, model, **kwargs)

    def on_examine_end(self, adversary, input, target, model, **kwargs):
        """Log gain and perturbation before advance() update the perturbation."""
        perturber = adversary.perturber

        # Check in gain and pert if gain > max_gain.
        idx_to_update = torch.where(adversary.gain > self.max_gain)[0]

        for i in idx_to_update:
            self.best_perturbation[i] = next(
                perturber.sub_perturbers[str(int(i))].parameters()
            ).detach()

        self.max_gain[idx_to_update] = adversary.gain[idx_to_update]

    def on_run_end(self, adversary, input, target, model, **kwargs):
        perturber = adversary.perturber

        idx_to_restore = torch.where(adversary.gain < self.max_gain)[0]

        for i in idx_to_restore:
            # Restore best_pert instead of latest_pert.
            param = next(perturber.sub_perturbers[str(int(i))].parameters())
            param.data = self.best_perturbation[i].data
