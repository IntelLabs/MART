#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import pytest
import torch

from mart.attack.initializer import Constant, Uniform, UniformLp


def test_constant_initializer(perturbation):
    constant = 1
    initializer = Constant(constant)
    initializer(perturbation)
    expected_perturbation = torch.ones(perturbation.shape)
    torch.testing.assert_close(perturbation, expected_perturbation)


def test_uniform_initializer(perturbation):
    min = 0
    max = 100

    initializer = Uniform(min, max)
    initializer(perturbation)

    assert torch.max(perturbation) <= max
    assert torch.min(perturbation) >= min


@pytest.mark.parametrize("p", [1, torch.inf])
def test_uniform_lp_initializer(p, perturbation):
    eps = 10

    initializer = UniformLp(eps, p)
    initializer(perturbation)

    assert torch.max(perturbation) <= eps
    assert torch.min(perturbation) >= -eps
