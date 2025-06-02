#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from unittest.mock import Mock

import pytest

from mart.attack import Perturber


def test_forward(input_data, target_data):
    initializer = Mock()
    projector = Mock()

    perturber = Perturber(initializer=initializer, projector=projector)

    perturber.configure_perturbation(input_data)

    output = perturber(input=input_data, target=target_data)

    initializer.assert_called_once()
    projector.assert_called_once()


def test_misconfiguration(input_data, target_data):
    initializer = Mock()
    projector = Mock()

    perturber = Perturber(initializer=initializer, projector=projector)

    with pytest.raises(RuntimeError):
        perturber(input=input_data, target=target_data)

    with pytest.raises(RuntimeError):
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
    for param in perturber.parameters():
        assert param.requires_grad
