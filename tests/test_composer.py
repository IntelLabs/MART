#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch

from mart.attack.composer import Additive, Composer, Overlay


def test_additive_method_forward(input_data, target_data, perturbation):
    method = Additive()

    output = method(perturbation, input=input_data, target=target_data)
    expected_output = input_data + perturbation
    torch.testing.assert_close(output, expected_output, equal_nan=True)


def test_overlay_method_forward(input_data, target_data, perturbation):
    method = Overlay()

    output = method(perturbation, input=input_data, target=target_data)
    mask = target_data["perturbable_mask"]
    mask = mask.to(input_data)
    expected_output = input_data * (1 - mask) + perturbation
    torch.testing.assert_close(output, expected_output, equal_nan=True)


def test_non_modality_composer_forward(input_data, target_data, perturbation):
    composer = Composer(method=Additive())

    # tensor input
    pert = perturbation
    input = input_data
    target = target_data
    output = composer(pert, input=input, target=target)
    expected_output = input_data + perturbation
    torch.testing.assert_close(output, expected_output, equal_nan=True)

    # tuple of tensor input
    pert = (perturbation,)
    input = (input_data,)
    target = (target_data,)
    output = composer(pert, input=input, target=target)
    expected_output = (input_data + perturbation,)
    assert type(output) == type(expected_output)
    torch.testing.assert_close(output[0], expected_output[0], equal_nan=True)


def test_modality_composer_forward(input_data, target_data, perturbation):
    input = {"rgb": input_data, "depth": input_data}
    target = target_data
    pert = {"rgb": perturbation, "depth": perturbation}

    rgb_method = Additive()
    depth_method = Overlay()
    composer = Composer(rgb=rgb_method, depth=depth_method)

    # dict
    output = composer(pert, input=input, target=target)
    expected_output = {
        "rgb": rgb_method(perturbation, input=input_data, target=target_data),
        "depth": depth_method(perturbation, input=input_data, target=target_data),
    }
    assert type(output) == type(expected_output)
    torch.testing.assert_close(output["rgb"], expected_output["rgb"], equal_nan=True)
    torch.testing.assert_close(output["depth"], expected_output["depth"], equal_nan=True)

    # list of dict
    output = composer([pert], input=[input], target=[target])
    expected_output = [
        {
            "rgb": rgb_method(perturbation, input=input_data, target=target_data),
            "depth": depth_method(perturbation, input=input_data, target=target_data),
        }
    ]
    assert type(output) == type(expected_output)
    torch.testing.assert_close(output[0]["rgb"], expected_output[0]["rgb"], equal_nan=True)
    torch.testing.assert_close(output[0]["depth"], expected_output[0]["depth"], equal_nan=True)

    # tuple of dict
    output = composer((pert,), input=(input,), target=(target,))
    expected_output = (
        {
            "rgb": rgb_method(perturbation, input=input_data, target=target_data),
            "depth": depth_method(perturbation, input=input_data, target=target_data),
        },
    )
    assert type(output) == type(expected_output)
    torch.testing.assert_close(output[0]["rgb"], expected_output[0]["rgb"], equal_nan=True)
    torch.testing.assert_close(output[0]["depth"], expected_output[0]["depth"], equal_nan=True)
