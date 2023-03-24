#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import importlib
from functools import partial
from unittest.mock import Mock, patch

import pytest
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException

import mart
from mart.attack.adversary import Adversary
from mart.attack.perturber import Perturber


def test_configure_perturbation(input_data):
    initializer = Mock()
    projector = Mock()
    composer = Mock()
    gain = Mock()

    perturber = Perturber(
        initializer=initializer, optimizer=None, composer=composer, projector=projector, gain=gain
    )

    perturber.configure_perturbation(input_data)

    initializer.assert_called_once()
    projector.assert_not_called()
    composer.assert_not_called()
    gain.assert_not_called()


def test_forward(input_data, target_data):
    initializer = mart.attack.initializer.Constant(1337)
    projector = Mock()
    composer = mart.attack.composer.Additive()
    gain = Mock()

    perturber = Perturber(
        initializer=initializer, optimizer=None, composer=composer, projector=projector, gain=gain
    )

    perturber.configure_perturbation(input_data)

    for _ in range(2):
        output_data = perturber(input=input_data, target=target_data)

        torch.testing.assert_close(output_data, input_data + 1337)

    # perturber needs to project and compose perturbation on every call
    assert projector.call_count == 2
    gain.assert_not_called()


def test_forward_fails(input_data, target_data):
    initializer = mart.attack.initializer.Constant(1337)
    projector = Mock()
    composer = mart.attack.composer.Additive()
    gain = Mock()

    perturber = Perturber(
        initializer=initializer, optimizer=None, composer=composer, projector=projector, gain=gain
    )

    with pytest.raises(MisconfigurationException):
        output_data = perturber(input=input_data, target=target_data)


def test_configure_optimizers(input_data, target_data):
    initializer = mart.attack.initializer.Constant(1337)
    optimizer = Mock()
    projector = Mock()
    composer = mart.attack.composer.Additive()
    gain = Mock()

    perturber = Perturber(
        initializer=initializer,
        optimizer=optimizer,
        composer=composer,
        projector=projector,
        gain=gain,
    )

    perturber.configure_perturbation(input_data)

    for _ in range(2):
        perturber.configure_optimizers()
        perturber(input=input_data, target=target_data)

    assert optimizer.call_count == 2
    assert projector.call_count == 2
    gain.assert_not_called()


def test_configure_optimizers_fails():
    initializer = mart.attack.initializer.Constant(1337)
    optimizer = Mock()
    projector = Mock()
    composer = mart.attack.composer.Additive()
    gain = Mock()

    perturber = Perturber(
        initializer=initializer,
        optimizer=optimizer,
        composer=composer,
        projector=projector,
        gain=gain,
    )

    with pytest.raises(MisconfigurationException):
        perturber.configure_optimizers()


def test_optimizer_parameters_with_gradient(input_data, target_data):
    initializer = mart.attack.initializer.Constant(1337)
    optimizer = partial(torch.optim.SGD, lr=0)
    projector = Mock()
    composer = mart.attack.composer.Additive()
    gain = Mock()

    perturber = Perturber(
        initializer=initializer,
        optimizer=optimizer,
        composer=composer,
        projector=projector,
        gain=gain,
    )

    perturber.configure_perturbation(input_data)
    opt = perturber.configure_optimizers()

    # Make sure each parameter in optimizer requires a gradient
    for param_group in opt.param_groups:
        for param in param_group["params"]:
            assert param.requires_grad


def test_training_step(input_data, target_data):
    initializer = mart.attack.initializer.Constant(1337)
    optimizer = Mock()
    projector = Mock()
    composer = mart.attack.composer.Additive()
    gain = Mock(return_value=torch.tensor(1337))
    model = Mock(return_value={})

    perturber = Perturber(
        initializer=initializer,
        optimizer=optimizer,
        composer=composer,
        projector=projector,
        gain=gain,
    )

    output = perturber.training_step(
        {"input": input_data, "target": target_data, "model": model}, 0
    )

    gain.assert_called_once()
    assert output == 1337


def test_training_step_with_many_gain(input_data, target_data):
    initializer = mart.attack.initializer.Constant(1337)
    optimizer = Mock()
    projector = Mock()
    composer = mart.attack.composer.Additive()
    gain = Mock(return_value=torch.tensor([1234, 5678]))
    model = Mock(return_value={})

    perturber = Perturber(
        initializer=initializer,
        optimizer=optimizer,
        composer=composer,
        projector=projector,
        gain=gain,
    )

    output = perturber.training_step(
        {"input": input_data, "target": target_data, "model": model}, 0
    )

    assert output == 1234 + 5678


def test_training_step_with_objective(input_data, target_data):
    initializer = mart.attack.initializer.Constant(1337)
    optimizer = Mock()
    projector = Mock()
    composer = mart.attack.composer.Additive()
    gain = Mock(return_value=torch.tensor([1234, 5678]))
    model = Mock(return_value={})
    objective = Mock(return_value=torch.tensor([True, False], dtype=torch.bool))

    perturber = Perturber(
        initializer=initializer,
        optimizer=optimizer,
        composer=composer,
        projector=projector,
        objective=objective,
        gain=gain,
    )

    output = perturber.training_step(
        {"input": input_data, "target": target_data, "model": model}, 0
    )

    assert output == 5678

    objective.assert_called_once()


def test_configure_gradient_clipping():
    initializer = mart.attack.initializer.Constant(1337)
    projector = Mock()
    composer = mart.attack.composer.Additive()
    optimizer = Mock(param_groups=[{"params": Mock()}, {"params": Mock()}])
    gradient_modifier = Mock()
    gain = Mock()

    perturber = Perturber(
        optimizer=optimizer,
        gradient_modifier=gradient_modifier,
        initializer=None,
        composer=None,
        projector=None,
        gain=gain,
    )
    # We need to mock a trainer since LightningModule does some checks
    perturber.trainer = Mock(gradient_clip_val=1.0, gradient_clip_algorithm="norm")

    perturber.configure_gradient_clipping(optimizer, 0)

    # Once for each parameter in the optimizer
    assert gradient_modifier.call_count == 2
