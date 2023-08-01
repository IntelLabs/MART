#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Modify a model so that it is convenient to attack.
# Common issues:
#     1. Make the model accept a single argument `output=model(batch)`;
#     2. Make the model return loss in eval mode;
#     3. Change non-differentiable operations.


class ListInputAsArgs:
    """Make a model expand input as non-keyword arguments."""

    def __call__(self, model):
        def forward(batch):
            return model(*batch)

        return forward


class Extract:
    """Example use case: extract the PyTorch model from an ART Estimator."""

    def __init__(self, attrib):
        self.attrib = attrib

    def __call__(self, model):
        model = getattr(model, self.attrib)
        return model
