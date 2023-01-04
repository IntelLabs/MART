#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import tqdm

from .base import Callback

__all__ = ["ProgressBar"]


class ProgressBar(Callback):
    """Display progress bar of attack iterations with the gain value."""

    def on_run_start(self, input, target, model, **kwargs):
        self.pbar = tqdm.tqdm(
            total=self.adversary.max_iters, leave=False, desc="Attack", unit="iter"
        )

    def on_examine_end(self, input, target, model, **kwargs):
        msg = ""
        if hasattr(self.adversary, "found"):
            # there is no adversary.found if adversary.objective_fn() is not defined.
            msg += f"found={int(sum(self.adversary.found))}/{len(input)}, "

        msg += f"avg_gain={float(self.adversary.gain.mean()):.2f}, "

        self.pbar.set_description(msg)
        self.pbar.update(1)

    def on_run_end(self, input, target, model, **kwargs):
        self.pbar.close()
        del self.pbar
