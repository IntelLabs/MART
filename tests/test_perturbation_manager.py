from typing import Iterable
from unittest.mock import Mock

import torch

from mart.attack.initializer import Constant
from mart.attack.perturber import Perturber


def test_perturbation_tensor_to_param_groups():
    input_data = torch.tensor([1.0, 2.0])
    initializer = Constant(constant=0)

    perturber = Perturber(initializer=initializer, optimizer=Mock(), composer=Mock(), gain=Mock())

    perturber.configure_perturbation(input_data)
    pert = perturber.perturbation
    assert isinstance(pert, torch.Tensor)
    assert pert.shape == pert.shape
    assert (pert == 0).all()

    param_groups = perturber.parameter_groups
    assert isinstance(param_groups, Iterable)
    assert param_groups[0]["params"].requires_grad


def test_perturbation_dict_to_param_groups():
    input_data = {"rgb": torch.tensor([1.0, 2.0]), "depth": torch.tensor([1.0, 2.0])}
    initializer = {"rgb": Constant(constant=0), "depth": Constant(constant=1)}
    perturber = Perturber(initializer=initializer, optimizer=Mock(), composer=Mock(), gain=Mock())

    perturber.configure_perturbation(input_data)
    pert = perturber.perturbation
    assert isinstance(pert, dict)
    assert (pert["rgb"] == 0).all()
    assert (pert["depth"] == 1).all()

    param_groups = perturber.parameter_groups
    assert len(param_groups) == 2
    param_groups = list(param_groups)
    assert param_groups[0]["params"].requires_grad
    # assert (param_groups[0]["params"] == 0).all()


def test_perturbation_tuple_dict_to_param_groups():
    input_data = (
        {"rgb": torch.tensor([1.0, 2.0]), "depth": torch.tensor([3.0, 4.0])},
        {"rgb": torch.tensor([-1.0, -2.0]), "depth": torch.tensor([-3.0, -4.0])},
    )
    initializer = {"rgb": Constant(constant=0), "depth": Constant(constant=1)}
    perturber = Perturber(initializer=initializer, optimizer=Mock(), composer=Mock(), gain=Mock())

    perturber.configure_perturbation(input_data)
    pert = perturber.perturbation
    assert isinstance(pert, tuple)
    assert (pert[0]["rgb"] == 0).all()
    assert (pert[0]["depth"] == 1).all()
    assert (pert[1]["rgb"] == 0).all()
    assert (pert[1]["depth"] == 1).all()

    param_groups = perturber.parameter_groups
    assert len(param_groups) == 4
    param_groups = list(param_groups)
    assert param_groups[0]["params"].requires_grad
