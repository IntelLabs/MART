#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import torch

from mart.attack.composer import (
    Additive,
    Composer,
    Mask,
    Overlay,
    RectangleCrop,
    RectanglePad,
    RectanglePerspectiveTransform,
)


def test_additive_composer_forward(input_data, target_data, perturbation):
    composer = Composer(functions={"additive": Additive()})

    output = composer(perturbation, input=input_data, target=target_data)
    expected_output = input_data + perturbation
    torch.testing.assert_close(output, expected_output, equal_nan=True)


def test_overlay_composer_forward(input_data, target_data, perturbation):
    composer = Composer(functions={"overlay": Overlay()})

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

    composer = Composer(functions={"mask": Mask(order=0), "additive": Additive(order=1)})
    output = composer(perturbation, input=input, target=target)
    torch.testing.assert_close(output, expected_output, equal_nan=True)


def test_rect_crop():
    key = "patch_coords"
    input = torch.zeros((3, 10, 10))
    perturbation = torch.ones_like(input)
    fn = RectangleCrop(coords_key=key)

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


def test_rect_pad():
    coords_key = "patch_coords"
    rect_coords_key = "rect_coords"

    rect_patch = torch.ones(3, 5, 5)
    patch_coords = torch.tensor(((0, 0), (5, 0), (5, 5), (5, 0)))

    input = torch.zeros((3, 10, 10))
    target = {coords_key: patch_coords}

    fn = RectanglePad(coords_key=coords_key, rect_coords_key=rect_coords_key)
    pert_padded, _input, _target = fn(rect_patch, input, target)

    pert_padded_expected = torch.zeros_like(input)
    pert_padded_expected[:, :5, :5] = 1

    assert torch.equal(pert_padded_expected, pert_padded)

    rect_coords_expected = [[0, 0], [5, 0], [5, 5], [0, 5]]
    assert _target[rect_coords_key] == rect_coords_expected


def test_rect_perspective_transform():
    coords_key = "patch_coords"
    rect_coords_key = "rect_coords"

    rect_coords = [[0, 0], [5, 0], [5, 5], [0, 5]]
    # Move from top left to bottom right.
    patch_coords = torch.tensor(((5, 5), (10, 5), (10, 10), (5, 10)))
    target = {coords_key: patch_coords, rect_coords_key: rect_coords}

    input = torch.zeros((3, 10, 10))

    pert_padded = torch.zeros_like(input)
    pert_padded[:, :5, :5] = 1

    fn = RectanglePerspectiveTransform(coords_key=coords_key, rect_coords_key=rect_coords_key)
    pert_coords, _input, _target = fn(pert_padded, input, target)
    pert_coords_expected = torch.zeros_like(input)
    pert_coords_expected[:, 5:, 5:] = 1
    # rounding numeric error from the perspective transformation.
    assert torch.equal(pert_coords.round(), pert_coords_expected)
