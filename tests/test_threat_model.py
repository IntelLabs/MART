#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import importlib
from unittest.mock import Mock, patch

import pytest
import torch

from mart.attack.threat_model import Additive, Overlay


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
