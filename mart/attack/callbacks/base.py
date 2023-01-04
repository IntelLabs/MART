#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import abc

from ..adversary import Adversary

__all__ = ["Callback"]


class Callback(abc.ABC):
    """Abstract base class of callbacks."""

    _adversary = None

    @property
    def adversary(self) -> Adversary:
        return self._adversary

    @adversary.setter
    def adversary(self, adversary: Adversary) -> None:
        self._adversary = adversary

    def on_run_start(self, input, target, model, **kwargs):
        pass

    def on_examine_start(self, input, target, model, **kwargs):
        pass

    def on_examine_end(self, input, target, model, **kwargs):
        pass

    def on_advance_start(self, input, target, model, **kwargs):
        pass

    def on_advance_end(self, input, target, model, **kwargs):
        pass

    def on_run_end(self, input, target, model, **kwargs):
        pass
