#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from unittest.mock import Mock

from PIL import Image, ImageChops
from torchvision.transforms import ToPILImage

from mart.attack import Adversary
from mart.attack.callbacks import PerturbedImageVisualizer


def test_visualizer_run_end(input_data, target_data, perturbation, tmp_path):
    folder = tmp_path / "test"
    input_list = [input_data]
    target_list = [target_data]

    # simulate an addition perturbation
    def perturb(input, target, model):
        result = [sample + perturbation for sample in input]
        return result

    model = Mock()
    adversary = Mock(spec=Adversary, side_effect=perturb)

    visualizer = PerturbedImageVisualizer(folder)
    visualizer.on_run_end(adversary, input_list, target_list, model)

    # verify that the visualizer created the JPG file
    expected_output_path = folder / target_data["file_name"]
    assert expected_output_path.exists()

    # verify image file content
    perturbed_img = input_data + perturbation
    converter = ToPILImage()
    expected_img = converter(perturbed_img / 255)
    expected_img.save(folder / "test_expected.jpg")

    stored_img = Image.open(expected_output_path)
    expected_stored_img = Image.open(folder / "test_expected.jpg")
    diff = ImageChops.difference(expected_stored_img, stored_img)
    assert not diff.getbbox()
