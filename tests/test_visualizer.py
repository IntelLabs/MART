#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import pytest

from mart.utils.imports import _HAS_TORCHVISION

if not _HAS_TORCHVISION:
    pytest.skip("test requires that torchvision is installed", allow_module_level=True)

from unittest.mock import Mock

from PIL import Image, ImageChops
from torchvision.transforms import ToPILImage

from mart.attack import Adversary
from mart.callbacks import PerturbedImageVisualizer


def test_visualizer_run_end(input_data, target_data, perturbation, tmp_path):
    folder = tmp_path / "test"

    # Convert to [0,255] space for adversary.
    perturbation_255 = perturbation * 255
    input_list = [input_data * 255]
    target_list = [target_data]

    # simulate an addition perturbation
    def perturb(input, target):
        result = [sample + perturbation_255 for sample in input]
        return result, target

    adversary = Mock(spec=Adversary, side_effect=perturb)
    trainer = Mock()
    outputs = Mock()
    target_model = Mock()

    # Canonical batch in Adversary.
    batch = (input_list, target_list, target_model)

    visualizer = PerturbedImageVisualizer(folder)
    visualizer.on_train_batch_end(trainer, adversary, outputs, batch, 0)
    visualizer.on_train_end(trainer, adversary)

    # verify that the visualizer created the JPG file
    expected_output_path = folder / target_data["file_name"]
    assert expected_output_path.exists()

    # verify image file content
    perturbed_img = input_data + perturbation
    converter = ToPILImage()
    expected_img = converter(perturbed_img)
    expected_img.save(folder / "test_expected.jpg")

    stored_img = Image.open(expected_output_path)
    expected_stored_img = Image.open(folder / "test_expected.jpg")
    diff = ImageChops.difference(expected_stored_img, stored_img)
    assert not diff.getbbox()
