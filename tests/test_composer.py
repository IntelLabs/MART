#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch

from mart.attack.composer import Additive, MaskAdditive, Overlay


def test_additive_composer_forward(input_data, target_data, perturbation):
    composer = Additive()

    output = composer(perturbation, input=input_data, target=target_data)
    expected_output = input_data + perturbation
    torch.testing.assert_close(output, expected_output, equal_nan=True)


def test_overlay_composer_forward(input_data, target_data, perturbation):
    composer = Overlay()

    output = composer(perturbation, input=input_data, target=target_data)
    mask = target_data["perturbable_mask"]
    mask = mask.to(input_data)
    expected_output = input_data * (1 - mask) + perturbation
    torch.testing.assert_close(output, expected_output, equal_nan=True)


def test_mask_additive_composer_forward():
    input = torch.zeros((2, 2))
    perturbation = torch.ones((2, 2))
    target = {"perturbable_mask": torch.eye(2)}
    expected_output = torch.eye(2)

    composer = MaskAdditive()
    output = composer(perturbation, input=input, target=target)
    torch.testing.assert_close(output, expected_output, equal_nan=True)
