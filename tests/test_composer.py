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
    modules = {"additive": Additive()}
    sequence = {"seq010": {"additive": ["perturbation", "input"]}}
    composer = Composer(perturber=perturber, modules=modules, sequence=sequence)

    output = composer(input=input_data, target=target_data)
    expected_output = input_data + perturbation
    torch.testing.assert_close(output, expected_output, equal_nan=True)


def test_overlay_composer_forward(input_data, target_data, perturbation):
    perturber = Mock(return_value=perturbation)
    modules = {"overlay": Overlay()}
    sequence = {"seq010": {"overlay": ["perturbation", "input", "target.perturbable_mask"]}}
    composer = Composer(perturber=perturber, modules=modules, sequence=sequence)

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
    modules = {"mask": Mask(), "additive": Additive()}
    sequence = {
        "seq010": {"mask": ["perturbation", "target.perturbable_mask"]},
        "seq020": {"additive": ["mask", "input"]},
    }
    composer = Composer(perturber=perturber, modules=modules, sequence=sequence)

    output = composer(input=input, target=target)
    torch.testing.assert_close(output, expected_output, equal_nan=True)
