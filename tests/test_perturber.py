#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from functools import partial
from unittest.mock import Mock

import pytest
import torch
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.optim import SGD

import mart
from mart.attack import Perturber


def test_forward(input_data, target_data):
    initializer = Mock()
    projector = Mock()

    perturber = Perturber(initializer=initializer, projector=projector)

    perturber.configure_perturbation(input_data)

    output = perturber(input=input_data, target=target_data)

    initializer.assert_called_once()
    projector.assert_called_once()

    # ouput should be broadcastable to input
    broadcast_shape = torch.broadcast_shapes(output.shape, input_data.shape)
    assert broadcast_shape == input_data.shape

def test_misconfiguration(input_data, target_data):
    initializer = Mock()
    projector = Mock()

    perturber = Perturber(initializer=initializer, projector=projector)

    with pytest.raises(MisconfigurationException):
        perturber(input=input_data, target=target_data)

    with pytest.raises(MisconfigurationException):
        perturber.parameters()


def test_configure_perturbation(input_data, target_data):
    initializer = Mock()
    projector = Mock()

    perturber = Perturber(initializer=initializer, projector=projector)

    perturber.configure_perturbation(input_data)
    perturber.configure_perturbation(input_data)
    perturber.configure_perturbation(input_data[:, :16, :16])

    # Each call to configure_perturbation should re-initialize the perturbation
    assert initializer.call_count == 3


def test_parameters(input_data, target_data):
    initializer = Mock()
    projector = Mock()

    perturber = Perturber(initializer=initializer, projector=projector)

    perturber.configure_perturbation(input_data)

    # Make sure each parameter in optimizer requires a gradient
    parameters = list(perturber.parameters())
    assert len(parameters) == 1
    for param in parameters:
        assert param.requires_grad


def test_shape(input_data, target_data):
    initializer = Mock()
    projector = Mock()
    shape = (None, 1, 1)

    perturber = Perturber(initializer=initializer, projector=projector, shape=shape)
    perturber.configure_perturbation(input_data)
    output = perturber(input=input_data, target=target_data)

    # ouput should be broadcastable to input
    broadcast_shape = torch.broadcast_shapes(output.shape, input_data.shape)
    assert broadcast_shape == input_data.shape


def test_bad_shape(input_data, target_data):
    initializer = Mock()
    projector = Mock()
    shape = (None, 2, 2)

    perturber = Perturber(initializer=initializer, projector=projector, shape=shape)
    perturber.configure_perturbation(input_data)
    output = perturber(input=input_data, target=target_data)

    # ouput should NOT be broadcastable to input
    with pytest.raises(RuntimeError):
       torch.broadcast_shapes(output.shape, input_data.shape)
