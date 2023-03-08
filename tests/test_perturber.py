#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import importlib
from unittest.mock import Mock, patch

import pytest
import torch

from mart.attack.perturber import Perturber


def test_perturber_repr(input_data, target_data):
    optimizer = Mock()
    initializer = Mock()
    gradient_modifier = Mock()
    projector = Mock()
    perturber = Perturber(optimizer, initializer, gradient_modifier, projector)

    # get additive perturber representation
    perturbation = torch.nn.UninitializedParameter()
    expected_repr = (
        f"{repr(perturbation)}, optimizer={optimizer}, initializer={initializer},"
        f"gradient_modifier={gradient_modifier}, projector={projector}"
    )
    representation = perturber.extra_repr()
    assert expected_repr == representation

    # generate again the perturber with an initialized
    # perturbation
    perturber.initialize_parameters(input_data, target_data)
    representation = perturber.extra_repr()
    assert expected_repr != representation


def test_perturber_forward(input_data, target_data):
    optimizer = Mock()
    initializer = Mock()
    perturber = Perturber(optimizer, initializer)

    output = perturber(input_data, target_data)
    expected_output = perturber.perturbation
    torch.testing.assert_close(output, expected_output, equal_nan=True)
