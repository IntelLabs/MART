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
    initializer = Mock()
    gradient_modifier = Mock()
    projector = Mock()
    perturber = Perturber(initializer, gradient_modifier, projector)

    # get additive perturber representation
    perturbation = torch.nn.UninitializedBuffer()
    expected_repr = (
        f"{repr(perturbation)}, initializer={initializer},"
        f"gradient_modifier={gradient_modifier}, projector={projector}"
    )
    representation = perturber.extra_repr()
    assert expected_repr == representation

    # generate again the perturber with an initialized
    # perturbation
    perturber.on_run_start(adversary=None, input=input_data, target=target_data, model=None)
    representation = perturber.extra_repr()
    assert expected_repr != representation


def test_perturber_forward(input_data, target_data):
    initializer = Mock()
    perturber = Perturber(initializer)

    perturber.on_run_start(adversary=None, input=input_data, target=target_data, model=None)
    output = perturber(input_data, target_data)
    expected_output = perturber.perturbation
    torch.testing.assert_close(output, expected_output, equal_nan=True)
