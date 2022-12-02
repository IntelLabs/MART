#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import abc

__all__ = ["Callback"]


class Callback(abc.ABC):
    """Abstract base class of callbacks."""

    def on_run_start(self, adversary, input, target, model, **kwargs):
        pass

    def on_examine_start(self, adversary, input, target, model, **kwargs):
        pass

    def on_examine_end(self, adversary, input, target, model, **kwargs):
        pass

    def on_advance_start(self, adversary, input, target, model, **kwargs):
        pass

    def on_advance_end(self, adversary, input, target, model, **kwargs):
        pass

    def on_run_end(self, adversary, input, target, model, **kwargs):
        pass
