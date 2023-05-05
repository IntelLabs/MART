#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from functools import partial
from unittest.mock import Mock

import torch
from torch.optim import SGD

import mart
from mart.attack import Adversary
from mart.attack.gradient_modifier import Sign
from mart.attack.perturber import Perturber


def test_adversary(input_data, target_data, perturbation):
    enforcer = Mock()
    perturber = Mock(return_value=input_data + perturbation)
    optimizer = Mock()
    max_iters = 3
    gain = Mock()

    adversary = Adversary(
        enforcer=enforcer,
        perturber=perturber,
        optimizer=optimizer,
        max_iters=max_iters,
        gain=gain,
    )

    output_data = adversary(input_data, target_data)

    optimizer.assert_not_called()
    gain.assert_not_called()
    perturber.assert_called_once()
    # The enforcer is only called when model is not None.
    enforcer.assert_not_called()
    torch.testing.assert_close(output_data, input_data + perturbation)


def test_adversary_with_model(input_data, target_data, perturbation):
    enforcer = Mock()
    initializer = Mock()
    perturber = Mock(return_value=input_data + perturbation)
    optimizer = Mock()
    max_iters = 3
    model = Mock(return_value={})
    gain = Mock(return_value=torch.tensor(0.0, requires_grad=True))

    adversary = Adversary(
        enforcer=enforcer,
        perturber=perturber,
        optimizer=optimizer,
        max_iters=3,
        gain=gain,
    )

    output_data = adversary(input_data, target_data, model=model)

    optimizer.assert_called_once()
    # The enforcer is only called when model is not None.
    enforcer.assert_called_once()
    # max_iters+1 because Adversary examines one last time
    assert gain.call_count == max_iters + 1
    assert model.call_count == max_iters + 1

    # Once with model=None to get perturbation.
    # When model=model, perturber.initialize_parameters() is called.
    assert perturber.call_count == 1

    torch.testing.assert_close(output_data, input_data + perturbation)


def test_adversary_perturber_hidden_params(input_data, target_data):
    initializer = Mock()
    composer = mart.attack.composer.Additive()
    perturber = Perturber(initializer=initializer, composer=composer)

    enforcer = Mock()
    optimizer = Mock()
    gain = Mock(return_value=torch.tensor(0.0, requires_grad=True))
    model = Mock(return_value={})

    adversary = Adversary(
        enforcer=enforcer,
        perturber=perturber,
        optimizer=optimizer,
        max_iters=1,
        gain=gain,
    )

    # Adversarial perturbation should not be updated by a regular training optimizer.
    params = [p for p in adversary.parameters()]
    assert len(params) == 0

    # Adversarial perturbation should not be saved to the model checkpoint.
    state_dict = adversary.state_dict()
    assert "perturber.perturbation" not in state_dict


def test_adversary_perturbation(input_data, target_data):
    composer = mart.attack.composer.Additive()
    enforcer = Mock()
    optimizer = partial(SGD, lr=1.0, maximize=True)

    def gain(logits):
        return logits.mean()

    # Perturbation initialized as zero.
    def initializer(x):
        torch.nn.init.constant_(x, 0)

    perturber = Perturber(initializer=initializer, composer=composer)

    adversary = Adversary(
        enforcer=enforcer,
        perturber=perturber,
        optimizer=optimizer,
        max_iters=1,
        gain=gain,
    )

    def model(input, target, model=None, **kwargs):
        return {"logits": adversary(input, target)}

    output1 = adversary(input_data.requires_grad_(), target_data, model=model)
    pert1 = perturber.perturbation.clone()
    output2 = adversary(input_data.requires_grad_(), target_data, model=model)
    pert2 = perturber.perturbation.clone()

    # The perturbation from multiple runs should be the same.
    torch.testing.assert_close(pert1, pert2)

    # Simulate a new batch of data of different size.
    new_input_data = torch.cat([input_data, input_data])
    output3 = adversary(new_input_data, target_data, model=model)


def test_adversary_gradient(input_data, target_data):
    composer = mart.attack.composer.Additive()
    enforcer = Mock()
    optimizer = partial(SGD, lr=1.0, maximize=True)

    # Force zeros, positive and negative gradients
    def gain(logits):
        return (
            (0 * logits[0, :, :]).mean()
            + (0.1 * logits[1, :, :]).mean()  # noqa: W503
            + (-0.1 * logits[2, :, :]).mean()  # noqa: W503
        )

    # Perturbation initialized as zero.
    def initializer(x):
        torch.nn.init.constant_(x, 0)

    perturber = Perturber(initializer=initializer, composer=composer)

    adversary = Adversary(
        enforcer=enforcer,
        perturber=perturber,
        optimizer=optimizer,
        max_iters=1,
        gain=gain,
        gradient_modifier=Sign(),
    )

    def model(input, target, model=None, **kwargs):
        return {"logits": adversary(input, target)}

    adversary(input_data, target_data, model=model)
    input_adv = adversary(input_data, target_data)

    perturbation = input_data - input_adv

    torch.testing.assert_close(perturbation.unique(), torch.Tensor([-1, 0, 1]))
