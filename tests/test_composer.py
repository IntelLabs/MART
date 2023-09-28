#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from unittest.mock import Mock

import torch

from mart.attack.composer import (
    Additive,
    Composer,
    Overlay,
    PerturbationMask,
    PerturbationRectangleCrop,
    PerturbationRectanglePad,
    PerturbationRectanglePerspectiveTransform,
)
from mart.utils import instantiate


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


def test_pert_mask_additive_composer_forward():
    input = torch.zeros((2, 2))
    perturbation = torch.ones((2, 2))
    target = {"perturbable_mask": torch.eye(2)}
    expected_output = torch.eye(2)

    perturber = Mock(return_value=perturbation)
    functions = {"pert_mask": PerturbationMask(order=0), "additive": Additive(order=1)}
    composer = Composer(perturber=perturber, functions=functions)

    output = composer(input=input, target=target)
    torch.testing.assert_close(output, expected_output, equal_nan=True)


def test_pert_rect_crop():
    key = "patch_coords"
    input = torch.zeros((3, 10, 10))
    perturbation = torch.ones_like(input)
    fn = PerturbationRectangleCrop(coords_key=key)

    # FIXME: four corner points (width, height) of a patch in the order of top-left, top-right, bottom-right, bottom-left.
    # A simple square patch.
    patch_coords = torch.tensor(((0, 0), (5, 0), (5, 5), (5, 0)))
    target = {key: patch_coords}

    rect_patch, _input, _target = fn(perturbation, input, target)
    assert torch.equal(input, _input)
    assert target == _target
    assert rect_patch.shape == (3, 5, 5)

    # A skew patch.
    patch_coords = torch.tensor(((1, 1), (5, 2), (7, 8), (3, 9)))
    target = {key: patch_coords}

    rect_patch, _input, _target = fn(perturbation, input, target)
    assert torch.equal(input, _input)
    assert target == _target
    assert rect_patch.shape == (3, 8, 6)


def test_pert_rect_pad():
    coords_key = "patch_coords"
    rect_coords_key = "rect_coords"

    rect_patch = torch.ones(3, 5, 5)
    patch_coords = torch.tensor(((0, 0), (5, 0), (5, 5), (5, 0)))

    input = torch.zeros((3, 10, 10))
    target = {coords_key: patch_coords}

    fn = PerturbationRectanglePad(coords_key=coords_key, rect_coords_key=rect_coords_key)
    pert_padded, _input, _target = fn(rect_patch, input, target)

    pert_padded_expected = torch.zeros_like(input)
    pert_padded_expected[:, :5, :5] = 1

    assert torch.equal(pert_padded_expected, pert_padded)

    rect_coords_expected = [[0, 0], [5, 0], [5, 5], [0, 5]]
    assert _target[rect_coords_key] == rect_coords_expected


def test_pert_rect_perspective_transform():
    coords_key = "patch_coords"
    rect_coords_key = "rect_coords"

    rect_coords = [[0, 0], [5, 0], [5, 5], [0, 5]]
    # Move from top left to bottom right.
    patch_coords = torch.tensor(((5, 5), (10, 5), (10, 10), (5, 10)))
    target = {coords_key: patch_coords, rect_coords_key: rect_coords}

    input = torch.zeros((3, 10, 10))

    pert_padded = torch.zeros_like(input)
    pert_padded[:, :5, :5] = 1

    fn = PerturbationRectanglePerspectiveTransform(
        coords_key=coords_key, rect_coords_key=rect_coords_key
    )
    pert_coords, _input, _target = fn(pert_padded, input, target)
    pert_coords_expected = torch.zeros_like(input)
    pert_coords_expected[:, 5:, 5:] = 1
    # rounding numeric error from the perspective transformation.
    assert torch.equal(pert_coords.round(), pert_coords_expected)


def test_rect_patch_additive_composer():
    overrides = ["+attack/composer=rect_patch_additive"]
    composer = instantiate(*overrides, export_node="attack.composer")

    input = torch.ones((3, 10, 10))
    perturbation = torch.ones_like(input) * 2

    input_adv_expected = input.clone()
    input_adv_expected[:, -5:, -5:] += 2

    # A simple square patch on the bottom right.
    patch_coords = torch.tensor(((5, 5), (10, 5), (10, 10), (5, 10)))
    perturbable_mask = torch.zeros((10, 10))
    perturbable_mask[-5:, -5:] = 1

    target = {"patch_coords": patch_coords, "perturbable_mask": perturbable_mask}
    input_adv = composer(perturbation, input=input, target=target)

    assert torch.allclose(input_adv_expected, input_adv)


def test_rect_patch_overlay_composer():
    overrides = ["+attack/composer=rect_patch_overlay"]
    composer = instantiate(*overrides, export_node="attack.composer")

    input = torch.ones((3, 10, 10))
    perturbation = torch.ones_like(input) * 2

    input_adv_expected = input.clone()
    input_adv_expected[:, -5:, -5:] = 2

    # A simple square patch on the bottom right.
    patch_coords = torch.tensor(((5, 5), (10, 5), (10, 10), (5, 10)))
    perturbable_mask = torch.zeros((10, 10))
    perturbable_mask[-5:, -5:] = 1

    target = {"patch_coords": patch_coords, "perturbable_mask": perturbable_mask}
    input_adv = composer(perturbation, input=input, target=target)

    assert torch.allclose(input_adv_expected, input_adv)
