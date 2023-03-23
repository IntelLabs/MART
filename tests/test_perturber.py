#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import importlib
from unittest.mock import Mock, patch

import pytest
import torch

import mart
from mart.attack.adversary import Adversary
from mart.attack.perturber import Perturber


def test_forward(input_data, target_data):
    initializer = mart.attack.initializer.Constant(1337)
    projector = Mock()
    composer = mart.attack.composer.Additive()

    perturber = Perturber(
        initializer=initializer, optimizer=None, composer=composer, projector=projector
    )

    for _ in range(2):
        output_data = perturber(input=input_data, target=target_data)

        torch.testing.assert_close(output_data, input_data + 1337)

    # perturber needs to project and compose perturbation on every call
    assert projector.call_count == 2


def test_configure_optimizers(input_data, target_data):
    initializer = mart.attack.initializer.Constant(1337)
    optimizer = Mock()
    projector = Mock()
    composer = mart.attack.composer.Additive()

    perturber = Perturber(
        initializer=initializer, optimizer=optimizer, composer=composer, projector=projector
    )

    for _ in range(2):
        perturber.configure_optimizers()
        perturber(input=input_data, target=target_data)

    assert optimizer.call_count == 2
    assert projector.call_count == 2


def test_training_step(input_data, target_data):
    initializer = mart.attack.initializer.Constant(1337)
    optimizer = Mock()
    projector = Mock()
    composer = mart.attack.composer.Additive()
    gain = Mock(shape=[])
    model = Mock(return_value={"loss": gain})

    perturber = Perturber(
        initializer=initializer, optimizer=optimizer, composer=composer, projector=projector
    )

    output = perturber.training_step(
        {"input": input_data, "target": target_data, "model": model}, 0
    )

    assert output == gain


def test_training_step_with_many_gain(input_data, target_data):
    initializer = mart.attack.initializer.Constant(1337)
    optimizer = Mock()
    projector = Mock()
    composer = mart.attack.composer.Additive()
    gain = torch.tensor([1234, 5678])
    model = Mock(return_value={"loss": gain})

    perturber = Perturber(
        initializer=initializer, optimizer=optimizer, composer=composer, projector=projector
    )

    output = perturber.training_step(
        {"input": input_data, "target": target_data, "model": model}, 0
    )

    assert output == gain.sum()


def test_training_step_with_objective(input_data, target_data):
    initializer = mart.attack.initializer.Constant(1337)
    optimizer = Mock()
    projector = Mock()
    composer = mart.attack.composer.Additive()
    gain = torch.tensor([1234, 5678])
    model = Mock(return_value={"loss": gain})
    objective = Mock(return_value=torch.tensor([True, False], dtype=torch.bool))

    perturber = Perturber(
        initializer=initializer,
        optimizer=optimizer,
        composer=composer,
        projector=projector,
        objective=objective,
    )

    output = perturber.training_step(
        {"input": input_data, "target": target_data, "model": model}, 0
    )

    assert output == gain[1]

    objective.assert_called_once()


def test_configure_gradient_clipping():
    initializer = mart.attack.initializer.Constant(1337)
    projector = Mock()
    composer = mart.attack.composer.Additive()
    optimizer = Mock(param_groups=[{"params": Mock()}, {"params": Mock()}])
    gradient_modifier = Mock()

    perturber = Perturber(
        optimizer=optimizer,
        gradient_modifier=gradient_modifier,
        initializer=None,
        composer=None,
        projector=None,
    )
    # We need to mock a trainer since LightningModule does some checks
    perturber.trainer = Mock(gradient_clip_val=1.0, gradient_clip_algorithm="norm")

    perturber.configure_gradient_clipping(optimizer, 0)

    # Once for each parameter in the optimizer
    assert gradient_modifier.call_count == 2
