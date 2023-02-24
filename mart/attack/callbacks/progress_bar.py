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

    def on_run_start(self, *, adversary, **kwargs):
        self.pbar = tqdm.tqdm(total=adversary.max_iters, leave=False, desc="Attack", unit="iter")

    def on_examine_end(self, *, adversary, **kwargs):
        input = kwargs["input"]
        msg = ""
        if hasattr(adversary, "found"):
            # there is no adversary.found if adversary.objective_fn() is not defined.
            msg += f"found={int(sum(adversary.found))}/{len(input)}, "

        msg += f"avg_gain={float(adversary.gain.mean()):.2f}, "

        self.pbar.set_description(msg)
        self.pbar.update(1)

    def on_run_end(self, **kwargs):
        self.pbar.close()
        del self.pbar
