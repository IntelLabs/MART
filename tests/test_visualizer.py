#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from unittest.mock import Mock

import pytest
import torch
from PIL import Image, ImageChops
from torchvision.transforms import ToPILImage

from mart.attack.callbacks import PerturbedImageVisualizer


def perturb(input, target, perturbation, **kwargs):
    result = [sample + perturbation for sample in input]
    return result


def test_visualizer_run_end(input_data, target_data, perturbation, tmp_path):
    folder = tmp_path / "test"
    input_list = [input_data]
    target_list = [target_data]

    model = Mock()
    adversary = Mock()
    adversary.perturber.return_value = perturbation
    adversary.threat_model.side_effect = perturb

    visualizer = PerturbedImageVisualizer(folder)
    visualizer.on_run_end(adversary, input_list, target_list, model)

    expected_output_path = folder / target_data["file_name"]
    assert expected_output_path.exists()
    adversary.perturber.assert_called_once_with(input_list, target_list)
    adversary.threat_model.assert_called_once_with(input_list, target_list, perturbation)

    # verify image file content
    perturbed_img = input_data + perturbation
    converter = ToPILImage()
    expected_img = converter(perturbed_img / 255)
    expected_img.save(folder / "test_expected.jpg")

    stored_img = Image.open(expected_output_path)
    expected_stored_img = Image.open(folder / "test_expected.jpg")
    diff = ImageChops.difference(expected_stored_img, stored_img)
    assert not diff.getbbox()
