#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from functools import partial
from unittest.mock import Mock

import pytest
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.optim import SGD

import mart
from mart.attack import Adversary, Composer, Perturber
from mart.attack.gradient_modifier import Sign


def test_adversary(input_data, target_data, perturbation):
    perturber = Mock(spec=Perturber, return_value=perturbation)
    composer = Mock(sepc=Composer, return_value=input_data + perturbation)
    gain = Mock()
    enforcer = Mock()

    adversary = Adversary(
        perturber=perturber,
        composer=composer,
        optimizer=None,
        gain=gain,
        enforcer=enforcer,
        max_iters=1,
    )

    output_data = adversary(input=input_data, target=target_data)

    # The enforcer and attacker should only be called when model is not None.
    perturber.assert_called_once()
    gain.assert_not_called()
    enforcer.assert_not_called()

    torch.testing.assert_close(output_data, input_data + perturbation)


def test_with_model(input_data, target_data, perturbation):
    perturber = Mock(spec=Perturber, return_value=perturbation)
    composer = Mock(sepc=Composer, return_value=input_data + perturbation)
    gain = Mock()
    enforcer = Mock()
    model = Mock(return_value={"loss": 0})
    sequence = Mock()
    optimizer = Mock()
    optimizer_fn = Mock(spec=mart.optim.OptimizerFactory, return_value=optimizer)

    adversary = Adversary(
        perturber=perturber,
        composer=composer,
        optimizer=optimizer_fn,
        gain=gain,
        enforcer=enforcer,
        max_iters=1,
    )

    output_data = adversary(input=input_data, target=target_data, model=model, sequence=sequence)

    # The enforcer is only called when model is not None.
    enforcer.assert_called_once()

    # Once with model=None to get perturbation.
    # When model=model, configure_perturbation() should be called.
    perturber.assert_called_once()
    assert gain.call_count == 2  # examine is called before done

    torch.testing.assert_close(output_data, input_data + perturbation)


def test_hidden_params(input_data, target_data, perturbation):
    initializer = Mock()
    composer = Mock()
    projector = Mock()

    perturber = Perturber(initializer=initializer, projector=projector)

    gain = Mock()
    enforcer = Mock()
    attacker = Mock(max_epochs=0, limit_train_batches=1, fit_loop=Mock(max_epochs=0))
    model = Mock()
    sequence = Mock()

    adversary = Adversary(
        perturber=perturber,
        composer=composer,
        optimizer=None,
        gain=gain,
        enforcer=enforcer,
        attacker=attacker,
    )

    # output_data = adversary(input=input_data, target=target_data, model=model, sequence=sequence)

    # Adversarial perturbation should not be updated by a regular training optimizer.
    params = [p for p in adversary.parameters()]
    assert len(params) == 0

    # Adversarial perturbation should not have any state dict items
    state_dict = adversary.state_dict()
    assert len(state_dict) == 0


def test_hidden_params_after_forward(input_data, target_data, perturbation):
    initializer = Mock()
    composer = Mock()
    projector = Mock()

    perturber = Perturber(initializer=initializer, projector=projector)

    gain = Mock()
    enforcer = Mock()
    model = Mock(return_value={"loss": 0})
    sequence = Mock()
    optimizer = Mock()
    optimizer_fn = Mock(return_value=optimizer)

    adversary = Adversary(
        perturber=perturber,
        composer=composer,
        optimizer=optimizer_fn,
        gain=gain,
        enforcer=enforcer,
        max_iters=1,
    )

    output_data = adversary(input=input_data, target=target_data, model=model, sequence=sequence)

    # Adversarial perturbation will have a perturbation after forward is called
    params = [p for p in adversary.parameters()]
    assert len(params) == 1

    # Adversarial perturbation should have a single state dict item
    state_dict = adversary.state_dict()
    assert len(state_dict) == 1


def test_perturbation(input_data, target_data, perturbation):
    perturber = Mock(spec=Perturber, return_value=perturbation)
    composer = Mock(spec=Composer, return_value=perturbation + input_data)
    gain = Mock()
    enforcer = Mock()
    model = Mock(return_value={"loss": 0})
    sequence = Mock()
    optimizer = Mock()
    optimizer_fn = Mock(spec=mart.optim.OptimizerFactory, return_value=optimizer)

    adversary = Adversary(
        perturber=perturber,
        composer=composer,
        optimizer=optimizer_fn,
        gain=gain,
        enforcer=enforcer,
        max_iters=1,
    )

    _ = adversary(input=input_data, target=target_data, model=model, sequence=sequence)
    output_data = adversary(input=input_data, target=target_data)

    # The enforcer is only called when model is not None.
    enforcer.assert_called_once()

    # Once with model and sequence and once without
    perturber.configure_perturbation.assert_called_once()
    assert perturber.call_count == 2

    torch.testing.assert_close(output_data, input_data + perturbation)


def test_forward_with_model(input_data, target_data):
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

    perturber = Perturber(
        initializer=initializer,
        projector=None,
    )

    adversary = Adversary(
        perturber=perturber,
        composer=composer,
        optimizer=optimizer,
        gain=gain,
        gradient_modifier=Sign(),
        enforcer=enforcer,
        max_iters=1,
    )

    def model(input, target, model=None, **kwargs):
        return {"logits": adversary(input=input, target=target)}

    sequence = Mock()

    adversary(input=input_data, target=target_data, model=model, sequence=sequence)
    input_adv = adversary(input=input_data, target=target_data)

    perturbation = input_data - input_adv

    torch.testing.assert_close(perturbation.unique(), torch.Tensor([-1, 0, 1]))
