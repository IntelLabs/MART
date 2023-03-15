#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import importlib
from unittest.mock import Mock, patch

import pytest
import torch

from mart.attack.threat_model import Additive, Integer, Lp, Overlay, Range


def test_additive_threat_model_forward(input_data, target_data, perturbation):
    threat_model = Additive()

    output = threat_model(input_data, target_data, perturbation)
    expected_output = input_data + perturbation
    torch.testing.assert_close(output, expected_output, equal_nan=True)


def test_overlay_threat_model_forward(input_data, target_data, perturbation):
    threat_model = Overlay()

    output = threat_model(input_data, target_data, perturbation)
    mask = target_data["perturbable_mask"]
    mask = mask.to(input_data)
    expected_output = input_data * (1 - mask) + perturbation
    torch.testing.assert_close(output, expected_output, equal_nan=True)


def test_constraint_range():
    constraint = Range(min=0, max=255)

    perturbation = torch.tensor([0, 128, 255])
    constraint(perturbation)

    with pytest.raises(Exception):
        perturbation = torch.tensor([-1, 255])
        constraint(perturbation)
        perturbation = torch.tensor([1, 256])
        constraint(perturbation)


def test_constraint_l2():
    constraint = Lp(eps=17.33, p=2, dim=[-1, -2, -3])

    # (3*10*10)**0.5 = 17.3205
    perturbation = torch.ones((3, 10, 10))
    # The same L2 norm, but a batch of data points.
    batch_perturbation = torch.stack((perturbation, perturbation))

    constraint(perturbation)
    with pytest.raises(Exception):
        constraint(perturbation * 2)

    constraint(batch_perturbation)
    with pytest.raises(Exception):
        constraint(batch_perturbation * 2)


def test_constraint_integer():
    constraint = Integer()
    perturbation = torch.tensor([1.0, 2.0])
    constraint(perturbation)

    perturbation = torch.tensor([1.0, 2.001])
    with pytest.raises(Exception):
        constraint(perturbation)
