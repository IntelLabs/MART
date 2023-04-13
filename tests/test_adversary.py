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
from mart.attack import Adversary
from mart.attack.gradient_modifier import Sign


def test_adversary(input_data, target_data, perturbation):
    initializer = Mock()
    projector = Mock()
    composer = Mock(return_value=perturbation + input_data)
    gain = Mock()
    enforcer = Mock()
    attacker = Mock(max_epochs=0, limit_train_batches=1, fit_loop=Mock(max_epochs=0))

    adversary = Adversary(
        initializer=initializer,
        optimizer=None,
        composer=composer,
        projector=projector,
        gain=gain,
        enforcer=enforcer,
        attacker=attacker,
    )

    adversary.configure_perturbation(input_data)

    output_data = adversary(input=input_data, target=target_data)

    # The enforcer and attacker should only be called when model is not None.
    enforcer.assert_not_called()
    attacker.fit.assert_not_called()
    assert attacker.fit_loop.max_epochs == 0

    initializer.assert_called_once()
    projector.assert_called_once()
    composer.assert_called_once()
    gain.assert_not_called()

    torch.testing.assert_close(output_data, input_data + perturbation)


def test_misconfiguration(input_data, target_data):
    initializer = mart.attack.initializer.Constant(1337)
    projector = Mock()
    composer = mart.attack.composer.Additive()
    gain = Mock()
    enforcer = Mock()
    attacker = Mock(max_epochs=0, limit_train_batches=1, fit_loop=Mock(max_epochs=0))

    adversary = Adversary(
        initializer=initializer,
        optimizer=None,
        composer=composer,
        projector=projector,
        gain=gain,
        enforcer=enforcer,
        attacker=attacker,
    )

    with pytest.raises(MisconfigurationException):
        output_data = adversary(input=input_data, target=target_data)


def test_with_model(input_data, target_data, perturbation):
    initializer = Mock()
    projector = Mock()
    composer = Mock(return_value=perturbation + input_data)
    gain = Mock()
    enforcer = Mock()
    attacker = Mock(max_epochs=0, limit_train_batches=1, fit_loop=Mock(max_epochs=0))

    adversary = Adversary(
        initializer=initializer,
        optimizer=None,
        composer=composer,
        projector=projector,
        gain=gain,
        enforcer=enforcer,
        attacker=attacker,
    )

    output_data = adversary(input=input_data, target=target_data, model=None, sequence=None)

    # The enforcer is only called when model is not None.
    enforcer.assert_called_once()
    attacker.fit.assert_called_once()

    # Once with model=None to get perturbation.
    # When model=model, configure_perturbation() should be called.
    initializer.assert_called_once()
    projector.assert_called_once()
    composer.assert_called_once()
    gain.assert_not_called()  # we mock attacker so this shouldn't be called

    torch.testing.assert_close(output_data, input_data + perturbation)


def test_hidden_params(input_data, target_data, perturbation):
    initializer = Mock()
    projector = Mock()
    composer = Mock(return_value=perturbation + input_data)
    gain = Mock()
    enforcer = Mock()
    attacker = Mock(max_epochs=0, limit_train_batches=1, fit_loop=Mock(max_epochs=0))

    adversary = Adversary(
        initializer=initializer,
        optimizer=None,
        composer=composer,
        projector=projector,
        gain=gain,
        enforcer=enforcer,
        attacker=attacker,
    )

    output_data = adversary(input=input_data, target=target_data, model=None, sequence=None)

    # Adversarial perturbation should not be updated by a regular training optimizer.
    params = [p for p in adversary.parameters()]
    assert len(params) == 0

    # Adversarial perturbation should not have any state dict items
    state_dict = adversary.state_dict()
    assert len(state_dict) == 0


def test_perturbation(input_data, target_data, perturbation):
    initializer = Mock()
    projector = Mock()
    composer = Mock(return_value=perturbation + input_data)
    gain = Mock()
    enforcer = Mock()
    attacker = Mock(max_epochs=0, limit_train_batches=1, fit_loop=Mock(max_epochs=0))

    adversary = Adversary(
        initializer=initializer,
        optimizer=None,
        composer=composer,
        projector=projector,
        gain=gain,
        enforcer=enforcer,
        attacker=attacker,
    )

    _ = adversary(input=input_data, target=target_data, model=None, sequence=None)
    output_data = adversary(input=input_data, target=target_data)

    # The enforcer is only called when model is not None.
    enforcer.assert_called_once()
    attacker.fit.assert_called_once()

    # Once with model and sequence and once without
    assert composer.call_count == 2

    torch.testing.assert_close(output_data, input_data + perturbation)


def test_gradient(input_data, target_data):
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

    adversary = Adversary(
        initializer=initializer,
        optimizer=optimizer,
        composer=composer,
        gain=gain,
        gradient_modifier=Sign(),
        enforcer=enforcer,
        max_iters=1,
    )

    def model(input, target, model=None, **kwargs):
        return {"logits": adversary(input=input, target=target)}

    adversary(input=input_data, target=target_data, model=model, sequence=None)
    input_adv = adversary(input=input_data, target=target_data)

    perturbation = input_data - input_adv

    torch.testing.assert_close(perturbation.unique(), torch.Tensor([-1, 0, 1]))


def test_configure_optimizers(input_data, target_data):
    initializer = mart.attack.initializer.Constant(1337)
    optimizer = Mock()
    projector = Mock()
    composer = mart.attack.composer.Additive()
    gain = Mock()

    adversary = Adversary(
        initializer=initializer,
        optimizer=optimizer,
        composer=composer,
        projector=projector,
        gain=gain,
    )

    adversary.configure_perturbation(input_data)

    for _ in range(2):
        adversary.configure_optimizers()
        adversary(input=input_data, target=target_data)

    assert optimizer.call_count == 2
    assert projector.call_count == 2
    gain.assert_not_called()


def test_configure_optimizers_fails():
    initializer = mart.attack.initializer.Constant(1337)
    optimizer = Mock()
    projector = Mock()
    composer = mart.attack.composer.Additive()
    gain = Mock()

    adversary = Adversary(
        initializer=initializer,
        optimizer=optimizer,
        composer=composer,
        projector=projector,
        gain=gain,
    )

    with pytest.raises(MisconfigurationException):
        adversary.configure_optimizers()


def test_optimizer_parameters_with_gradient(input_data, target_data):
    initializer = mart.attack.initializer.Constant(1337)
    optimizer = partial(torch.optim.SGD, lr=0)
    projector = Mock()
    composer = mart.attack.composer.Additive()
    gain = Mock()

    adversary = Adversary(
        initializer=initializer,
        optimizer=optimizer,
        composer=composer,
        projector=projector,
        gain=gain,
    )

    adversary.configure_perturbation(input_data)
    opt = adversary.configure_optimizers()

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

    adversary = Adversary(
        initializer=initializer,
        optimizer=optimizer,
        composer=composer,
        projector=projector,
        gain=gain,
    )

    output = adversary.training_step(
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

    adversary = Adversary(
        initializer=initializer,
        optimizer=optimizer,
        composer=composer,
        projector=projector,
        gain=gain,
    )

    output = adversary.training_step(
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

    adversary = Adversary(
        initializer=initializer,
        optimizer=optimizer,
        composer=composer,
        projector=projector,
        objective=objective,
        gain=gain,
    )

    output = adversary.training_step(
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

    adversary = Adversary(
        optimizer=optimizer,
        gradient_modifier=gradient_modifier,
        initializer=None,
        composer=None,
        projector=None,
        gain=gain,
    )
    # We need to mock a trainer since LightningModule does some checks
    adversary.trainer = Mock(gradient_clip_val=1.0, gradient_clip_algorithm="norm")

    adversary.configure_gradient_clipping(optimizer, 0)

    # Once for each parameter in the optimizer
    assert gradient_modifier.call_count == 2
