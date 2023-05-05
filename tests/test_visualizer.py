#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
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
    def perturb(input):
        result = [sample + perturbation for sample in input]
        return result

    trainer = Mock()
    model = Mock(return_value=perturb(input_list))
    outputs = Mock()
    batch = {"input": input_list, "target": target_list}
    adversary = Mock(spec=Adversary, side_effect=perturb)

    visualizer = PerturbedImageVisualizer(folder)
    visualizer.on_train_batch_end(trainer, model, outputs, batch, 0)
    visualizer.on_train_end(trainer, model)

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
