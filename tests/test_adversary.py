#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from unittest.mock import Mock

import pytest
import torch

import mart
from mart.attack import Adversary, NoAdversary
from mart.attack.perturber import Perturber


def test_no_adversary(input_data, target_data):
    adversary = NoAdversary()

    # Not having a model should not change the output.
    output_data = adversary(input_data, target_data)

    torch.testing.assert_close(output_data, input_data)


def test_no_adversary_with_model(input_data, target_data):
    adversary = NoAdversary()
    model = Mock()

    # Having a model should not change the output.
    output_data = adversary(input_data, target_data, model=model)

    model.assert_not_called()
    torch.testing.assert_close(output_data, input_data)


def test_adversary(input_data, target_data, perturbation):
    threat_model = mart.attack.threat_model.Additive()
    perturber = Mock(return_value=perturbation)
    max_iters = 3
    gain = Mock()

    adversary = Adversary(threat_model=threat_model, perturber=perturber, max_iters=3, gain=gain)

    output_data = adversary(input_data, target_data)

    gain.assert_not_called()
    perturber.assert_called_once()
    torch.testing.assert_close(output_data, input_data + perturbation)


def test_adversary_with_model(input_data, target_data, perturbation):
    threat_model = mart.attack.threat_model.Additive()
    initializer = Mock()
    perturber = Mock(return_value=perturbation)
    max_iters = 3
    model = Mock(return_value={})
    gain = Mock(return_value=torch.tensor(0.0, requires_grad=True))

    adversary = Adversary(threat_model=threat_model, perturber=perturber, max_iters=3, gain=gain)

    output_data = adversary(input_data, target_data, model=model)

    # max_iters+1 because Adversary examines one last time
    assert gain.call_count == max_iters + 1
    assert model.call_count == max_iters + 1

    # The adversary only calls this once to get the perturbation
    assert perturber.call_count == 1

    torch.testing.assert_close(output_data, input_data + perturbation)


def test_adversary_perturber_hidden_params(input_data, target_data):
    initializer = Mock()
    optimizer = Mock()
    perturber = Perturber(optimizer, initializer)
    perturber(input_data, target_data)

    threat_model = mart.attack.threat_model.Additive()
    gain = Mock(return_value=torch.tensor(0.0, requires_grad=True))
    model = Mock(return_value={})

    adversary = Adversary(threat_model=threat_model, perturber=perturber, max_iters=1, gain=gain)
    output_data = adversary(input_data, target_data, model=model)

    # Adversarial perturbation should not be updated by a regular training optimizer.
    params = [p for p in adversary.parameters()]
    assert len(params) == 0

    # Adversarial perturbation should not be saved to the model checkpoint.
    state_dict = adversary.state_dict()
    assert "perturber.perturbation" not in state_dict
