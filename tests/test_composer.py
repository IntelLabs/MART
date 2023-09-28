#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from unittest.mock import Mock

import torch

from mart.attack.composer import Additive, Composer, Mask, Overlay


def test_additive_composer_forward(input_data, target_data, perturbation):
    perturber = Mock(return_value=perturbation)
    functions = {"additive": Additive()}
    composer = Composer(perturber=perturber, functions=functions)

    output = composer(input=input_data, target=target_data)
    expected_output = input_data + perturbation
    torch.testing.assert_close(output, expected_output, equal_nan=True)


def test_overlay_composer_forward(input_data, target_data, perturbation):
    perturber = Mock(return_value=perturbation)
    functions = {"overlay": Overlay()}
    composer = Composer(perturber=perturber, functions=functions)

    output = composer(input=input_data, target=target_data)
    mask = target_data["perturbable_mask"]
    mask = mask.to(input_data)
    expected_output = input_data * (1 - mask) + perturbation
    torch.testing.assert_close(output, expected_output, equal_nan=True)


def test_mask_additive_composer_forward():
    input = torch.zeros((2, 2))
    perturbation = torch.ones((2, 2))
    target = {"perturbable_mask": torch.eye(2)}
    expected_output = torch.eye(2)

    perturber = Mock(return_value=perturbation)
    functions = {"mask": Mask(order=0), "additive": Additive(order=1)}
    composer = Composer(perturber=perturber, functions=functions)

    output = composer(input=input, target=target)
    torch.testing.assert_close(output, expected_output, equal_nan=True)
