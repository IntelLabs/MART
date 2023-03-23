#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from functools import partial
from unittest.mock import Mock

import pytorch_lightning as pl
import torch
from torch.optim import SGD

import mart
from mart.attack import Adversary, Perturber


def test_adversary(input_data, target_data, perturbation):
    enforcer = Mock()
    perturber = Mock(return_value=perturbation + input_data)
    attacker = Mock(max_epochs=0, limit_train_batches=1, fit_loop=Mock(max_epochs=0))

    adversary = Adversary(
        enforcer=enforcer,
        perturber=perturber,
        attacker=attacker,
    )

    output_data = adversary(input=input_data, target=target_data)

    # The enforcer and attacker should only be called when model is not None.
    enforcer.assert_not_called()
    attacker.fit.assert_not_called()
    assert attacker.fit_loop.max_epochs == 0

    perturber.assert_called_once()

    torch.testing.assert_close(output_data, input_data + perturbation)


def test_adversary_with_model(input_data, target_data, perturbation):
    enforcer = Mock()
    perturber = Mock(return_value=input_data + perturbation)
    attacker = Mock(max_epochs=0, limit_train_batches=1, fit_loop=Mock(max_epochs=0))

    adversary = Adversary(
        enforcer=enforcer,
        perturber=perturber,
        attacker=attacker,
    )

    output_data = adversary(input=input_data, target=target_data, model=None, sequence=None)

    # The enforcer is only called when model is not None.
    enforcer.assert_called_once()
    attacker.fit.assert_called_once()

    # Once with model=None to get perturbation.
    # When model=model, perturber.initialize_parameters() is called.
    assert perturber.call_count == 1

    torch.testing.assert_close(output_data, input_data + perturbation)


def test_adversary_hidden_params(input_data, target_data, perturbation):
    enforcer = Mock()
    perturber = Mock(return_value=input_data + perturbation)
    attacker = Mock(max_epochs=0, limit_train_batches=1, fit_loop=Mock(max_epochs=0))

    adversary = Adversary(
        enforcer=enforcer,
        perturber=perturber,
        attacker=attacker,
    )

    output_data = adversary(input=input_data, target=target_data, model=None, sequence=None)

    # Adversarial perturbation should not be updated by a regular training optimizer.
    params = [p for p in adversary.parameters()]
    assert len(params) == 0

    # Adversarial perturbation should not have any state dict items
    state_dict = adversary.state_dict()
    assert len(state_dict) == 0


def test_adversary_perturbation(input_data, target_data, perturbation):
    enforcer = Mock()
    perturber = Mock(return_value=input_data + perturbation)
    attacker = Mock(max_epochs=0, limit_train_batches=1, fit_loop=Mock(max_epochs=0))

    adversary = Adversary(
        enforcer=enforcer,
        perturber=perturber,
        attacker=attacker,
    )

    _ = adversary(input=input_data, target=target_data, model=None, sequence=None)
    output_data = adversary(input=input_data, target=target_data)

    # The enforcer is only called when model is not None.
    enforcer.assert_called_once()
    attacker.fit.assert_called_once()

    # Once with model and sequence and once without
    assert perturber.call_count == 2

    torch.testing.assert_close(output_data, input_data + perturbation)
