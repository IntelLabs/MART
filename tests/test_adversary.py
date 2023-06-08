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
    composer = mart.attack.composer.Additive()
    gain = Mock()
    enforcer = Mock()
    attacker = Mock(max_epochs=0, limit_train_batches=1, fit_loop=Mock(max_epochs=0))

    adversary = Adversary(
        perturber=perturber,
        composer=composer,
        optimizer=None,
        gain=gain,
        enforcer=enforcer,
        attacker=attacker,
    )

    output_data = adversary(input=input_data, target=target_data)

    # The enforcer and attacker should only be called when model is not None.
    enforcer.assert_not_called()
    attacker.fit.assert_not_called()
    assert attacker.fit_loop.max_epochs == 0

    perturber.assert_called_once()
    gain.assert_not_called()

    torch.testing.assert_close(output_data, input_data + perturbation)


def test_with_model(input_data, target_data, perturbation):
    perturber = Mock(spec=Perturber, return_value=perturbation)
    composer = mart.attack.composer.Additive()
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

    output_data = adversary(input=input_data, target=target_data, model=model, sequence=sequence)

    # The enforcer is only called when model is not None.
    enforcer.assert_called_once()
    attacker.fit.assert_called_once()

    # Once with model=None to get perturbation.
    # When model=model, configure_perturbation() should be called.
    perturber.assert_called_once()
    gain.assert_not_called()  # we mock attacker so this shouldn't be called

    torch.testing.assert_close(output_data, input_data + perturbation)


def test_hidden_params(input_data, target_data, perturbation):
    initializer = Mock()
    composer = mart.attack.composer.Additive()
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
    composer = mart.attack.composer.Additive()
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

    output_data = adversary(input=input_data, target=target_data, model=model, sequence=sequence)

    # Adversarial perturbation will have one parameter after forward is called.
    params = [p for p in adversary.parameters()]
    assert len(params) == 1

    # Adversarial perturbation should have one state dict item being exported to the model checkpoint.
    state_dict = adversary.state_dict()
    assert len(state_dict) == 1


def test_loading_perturbation_from_state_dict():
    initializer = Mock()
    composer = mart.attack.composer.Additive()
    projector = Mock()

    perturber = Perturber(initializer=initializer, projector=projector)

    gain = Mock()
    enforcer = Mock()
    attacker = Mock(max_epochs=0, limit_train_batches=1, fit_loop=Mock(max_epochs=0))

    adversary = Adversary(
        perturber=perturber,
        composer=composer,
        optimizer=None,
        gain=gain,
        enforcer=enforcer,
        attacker=attacker,
    )

    # We should be able to load arbitrary state_dict, because Adversary ignores state_dict.
    # We want this behavior for Adversary because model checkpoints may include perturbation in state_dict
    # that is not loadable before initialization of perturbation.
    adversary.load_state_dict({"perturber.perturbation": torch.tensor([1.0, 2.0])})

    # Adversary ignores load_state_dict() quietly, so perturbation is still None.
    assert adversary.perturber.perturbation is None


def test_perturbation(input_data, target_data, perturbation):
    perturber = Mock(spec=Perturber, return_value=perturbation)
    composer = mart.attack.composer.Additive()
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

    _ = adversary(input=input_data, target=target_data, model=model, sequence=sequence)
    output_data = adversary(input=input_data, target=target_data)

    # The enforcer is only called when model is not None.
    enforcer.assert_called_once()
    attacker.fit.assert_called_once()

    # Once with model and sequence and once without
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


def test_configure_optimizers(input_data, target_data):
    perturber = Mock()
    composer = mart.attack.composer.Additive()
    optimizer = Mock(spec=mart.optim.OptimizerFactory)
    gain = Mock()

    adversary = Adversary(
        perturber=perturber,
        composer=composer,
        optimizer=optimizer,
        gain=gain,
    )

    adversary.configure_optimizers()

    assert optimizer.call_count == 1
    gain.assert_not_called()


def test_training_step(input_data, target_data):
    perturber = Mock()
    composer = mart.attack.composer.Additive()
    optimizer = Mock(spec=mart.optim.OptimizerFactory)
    gain = Mock(return_value=torch.tensor(1337))
    model = Mock(return_value={})

    adversary = Adversary(
        perturber=perturber,
        composer=composer,
        optimizer=optimizer,
        gain=gain,
    )

    output = adversary.training_step(
        {"input": input_data, "target": target_data, "model": model}, 0
    )

    gain.assert_called_once()
    assert output == 1337


def test_training_step_with_many_gain(input_data, target_data):
    perturber = Mock()
    composer = mart.attack.composer.Additive()
    optimizer = Mock(spec=mart.optim.OptimizerFactory)
    gain = Mock(return_value=torch.tensor([1234, 5678]))
    model = Mock(return_value={})

    adversary = Adversary(
        perturber=perturber,
        composer=composer,
        optimizer=optimizer,
        gain=gain,
    )

    output = adversary.training_step(
        {"input": input_data, "target": target_data, "model": model}, 0
    )

    assert output == 1234 + 5678


def test_training_step_with_objective(input_data, target_data):
    perturber = Mock()
    composer = mart.attack.composer.Additive()
    optimizer = Mock(spec=mart.optim.OptimizerFactory)
    gain = Mock(return_value=torch.tensor([1234, 5678]))
    model = Mock(return_value={})
    objective = Mock(return_value=torch.tensor([True, False], dtype=torch.bool))

    adversary = Adversary(
        perturber=perturber,
        composer=composer,
        optimizer=optimizer,
        objective=objective,
        gain=gain,
    )

    output = adversary.training_step(
        {"input": input_data, "target": target_data, "model": model}, 0
    )

    assert output == 5678

    objective.assert_called_once()


def test_configure_gradient_clipping():
    perturber = Mock()
    composer = mart.attack.composer.Additive()
    optimizer = Mock(
        spec=mart.optim.OptimizerFactory, param_groups=[{"params": Mock()}, {"params": Mock()}]
    )
    gradient_modifier = Mock()
    gain = Mock()

    adversary = Adversary(
        perturber=perturber,
        composer=composer,
        optimizer=optimizer,
        gradient_modifier=gradient_modifier,
        gain=gain,
    )
    # We need to mock a trainer since LightningModule does some checks
    adversary.trainer = Mock(gradient_clip_val=1.0, gradient_clip_algorithm="norm")

    adversary.configure_gradient_clipping(optimizer, 0)

    # Once for each parameter in the optimizer
    assert gradient_modifier.call_count == 2
