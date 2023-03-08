#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from functools import partial
from unittest.mock import Mock

import pytest
import torch
from torch.optim import SGD

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
    optimizer = Mock()
    max_iters = 3
    gain = Mock()

    adversary = Adversary(
        threat_model=threat_model, perturber=perturber, optimizer=optimizer, max_iters=3, gain=gain
    )

    output_data = adversary(input_data, target_data)

    optimizer.assert_not_called()
    gain.assert_not_called()
    perturber.assert_called_once()
    torch.testing.assert_close(output_data, input_data + perturbation)


def test_adversary_with_model(input_data, target_data, perturbation):
    threat_model = mart.attack.threat_model.Additive()
    initializer = Mock()
    parameter_groups = Mock(return_value=[])
    perturber = Mock(return_value=perturbation, parameter_groups=parameter_groups)
    optimizer = Mock()
    max_iters = 3
    model = Mock(return_value={})
    gain = Mock(return_value=torch.tensor(0.0, requires_grad=True))

    adversary = Adversary(
        threat_model=threat_model, perturber=perturber, optimizer=optimizer, max_iters=3, gain=gain
    )

    output_data = adversary(input_data, target_data, model=model)

    parameter_groups.assert_called_once()
    optimizer.assert_called_once()
    # max_iters+1 because Adversary examines one last time
    assert gain.call_count == max_iters + 1
    assert model.call_count == max_iters + 1

    # Once with model=None to get perturbation.
    # When model=model, perturber.initialize_parameters() is called.
    assert perturber.call_count == 1

    torch.testing.assert_close(output_data, input_data + perturbation)


def test_adversary_perturber_hidden_params(input_data, target_data):
    initializer = Mock()
    perturber = Perturber(initializer)

    threat_model = mart.attack.threat_model.Additive()
    optimizer = Mock()
    gain = Mock(return_value=torch.tensor(0.0, requires_grad=True))
    model = Mock(return_value={})

    adversary = Adversary(
        threat_model=threat_model, perturber=perturber, optimizer=optimizer, max_iters=1, gain=gain
    )
    output_data = adversary(input_data, target_data, model=model)

    # Adversarial perturbation should not be updated by a regular training optimizer.
    params = [p for p in adversary.parameters()]
    assert len(params) == 0

    # Adversarial perturbation should not be saved to the model checkpoint.
    state_dict = adversary.state_dict()
    assert "perturber.perturbation" not in state_dict


def test_adversary_perturbation(input_data, target_data):
    threat_model = mart.attack.threat_model.Additive()
    optimizer = partial(SGD, lr=1.0, maximize=True)

    def gain(logits):
        return logits.mean()

    # Perturbation initialized as zero.
    def initializer(x):
        torch.nn.init.constant_(x, 0)

    perturber = Perturber(initializer)

    adversary = Adversary(
        threat_model=threat_model, perturber=perturber, optimizer=optimizer, max_iters=1, gain=gain
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
