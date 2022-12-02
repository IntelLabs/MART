#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import torch

from .base import Objective

__all__ = ["Mispredict", "RandomTarget"]


class Mispredict(Objective):
    def __call__(self, input, target):
        # FIXME: I don't like this argmax call. It feels like this should receive input tensor of
        #        the same shape as target?
        mispredictions = input.argmax(dim=-1) != target
        return mispredictions


class RandomTarget(Objective):
    def __init__(self, nb_classes, gain_fn) -> None:
        self.nb_classes = nb_classes
        self.gain_fn = gain_fn

    def __call__(self, logits, target):
        # FIXME: It may be better if we make sure that the pseudo target is different from target.
        pseudo_target = torch.randint_like(target, low=0, high=self.nb_classes)
        return self.gain_fn(logits, pseudo_target)
