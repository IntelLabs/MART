from typing import Iterable

import torch

from mart.attack.initializer import Constant
from mart.attack.perturbation_manager import PerturbationManager


def test_perturbation_tensor():
    input_data = torch.tensor([1.0, 2.0])
    initializer = Constant(constant=0)

    pert_manager = PerturbationManager(initializer=initializer)

    pert = pert_manager(input_data)
    assert isinstance(pert, torch.Tensor)
    assert pert.shape == pert.shape
    assert (pert == 0).all()

    param_groups = pert_manager.parameter_groups
    assert isinstance(param_groups, Iterable)
    assert param_groups[0]["params"].requires_grad


def test_perturbation_dict():
    input_data = {"rgb": torch.tensor([1.0, 2.0]), "depth": torch.tensor([1.0, 2.0])}
    initializer = {"rgb": Constant(constant=0), "depth": Constant(constant=1)}
    pert_manager = PerturbationManager(initializer=initializer)

    pert = pert_manager(input_data)
    assert isinstance(pert, dict)
    assert (pert["rgb"] == 0).all()
    assert (pert["depth"] == 1).all()

    param_groups = pert_manager.parameter_groups
    assert len(param_groups) == 2
    param_groups = list(param_groups)
    assert param_groups[0]["params"].requires_grad
    # assert (param_groups[0]["params"] == 0).all()


def test_perturbation_tuple_dict():
    input_data = (
        {"rgb": torch.tensor([1.0, 2.0]), "depth": torch.tensor([3.0, 4.0])},
        {"rgb": torch.tensor([-1.0, -2.0]), "depth": torch.tensor([-3.0, -4.0])},
    )
    initializer = {"rgb": Constant(constant=0), "depth": Constant(constant=1)}
    pert_manager = PerturbationManager(initializer=initializer)

    pert = pert_manager(input_data)
    assert isinstance(pert, tuple)
    assert (pert[0]["rgb"] == 0).all()
    assert (pert[0]["depth"] == 1).all()
    assert (pert[1]["rgb"] == 0).all()
    assert (pert[1]["depth"] == 1).all()

    param_groups = pert_manager.parameter_groups
    assert len(param_groups) == 4
    param_groups = list(param_groups)
    assert param_groups[0]["params"].requires_grad
