#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch

from mart.attack.composer import Additive, Overlay


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
