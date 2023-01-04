#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from unittest.mock import Mock, patch

import pytest
import torch

from mart.attack.perturber import BatchPerturber, Perturber


@pytest.fixture(scope="function")
def perturber_batch():
    # function to mock perturbation
    def perturbation(input, target):
        return input + torch.ones(*input.shape)

    # setup batch mock
    perturber = Mock(name="perturber_mock", spec=Perturber, side_effect=perturbation)
    perturber_factory = Mock(return_value=perturber)

    batch = BatchPerturber(perturber_factory)

    return batch


@pytest.fixture(scope="function")
def input_data_batch():
    batch_size = 2
    image_size = (3, 32, 32)

    input_data = {}
    input_data["image_batch"] = torch.zeros(batch_size, *image_size)
    input_data["image_batch_list"] = [torch.zeros(*image_size) for _ in range(batch_size)]
    input_data["target"] = {"perturbable_mask": torch.ones(*image_size)}

    return input_data


def test_batch_run_start(perturber_batch, input_data_batch):
    assert isinstance(perturber_batch, BatchPerturber)

    # start perturber batch
    model = Mock()
    perturber_batch.on_run_start(
        input_data_batch["image_batch"], input_data_batch["target"], model
    )

    batch_size, _, _, _ = input_data_batch["image_batch"].shape
    assert len(perturber_batch.perturbers) == batch_size


def test_batch_forward(perturber_batch, input_data_batch):
    assert isinstance(perturber_batch, BatchPerturber)

    # start perturber batch
    model = Mock()
    perturber_batch.on_run_start(
        input_data_batch["image_batch"], input_data_batch["target"], model
    )

    perturbed_images = perturber_batch(input_data_batch["image_batch"], input_data_batch["target"])
    expected = torch.ones(*perturbed_images.shape)
    torch.testing.assert_close(perturbed_images, expected)


def test_tuple_batch_forward(perturber_batch, input_data_batch):
    assert isinstance(perturber_batch, BatchPerturber)

    # start perturber batch
    model = Mock()
    perturber_batch.on_run_start(
        input_data_batch["image_batch_list"], input_data_batch["target"], model
    )

    perturbed_images = perturber_batch(
        input_data_batch["image_batch_list"], input_data_batch["target"]
    )
    expected = [
        torch.ones(*input_data_batch["image_batch_list"][0].shape)
        for _ in range(len(input_data_batch["image_batch_list"]))
    ]

    for output, expected_output in zip(expected, perturbed_images):
        torch.testing.assert_close(output, expected_output)
