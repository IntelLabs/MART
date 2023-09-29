#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import pytest
import torch

from mart.attack.projector import Lp, Mask, Range


def test_range_projector_repr():
    min = 0
    max = 100
    quantize = True
    projector = Range(quantize, min, max)
    representation = repr(projector)
    expected_representation = (
        f"{projector.__class__.__name__}(quantize={quantize}, min={min}, max={max})"
    )
    assert representation == expected_representation


@pytest.mark.parametrize("quantize", [False, True])
@pytest.mark.parametrize("min", [-10, 0, 10])
@pytest.mark.parametrize("max", [10, 100, 110])
def test_range_projector(quantize, min, max, input_data, target_data, perturbation):
    projector = Range(quantize, min, max)
    projector(perturbation, input=input_data, target=target_data)

    assert torch.max(perturbation) <= max
    assert torch.min(perturbation) >= min


@pytest.mark.parametrize("eps", [30, 40, 50, 60])
@pytest.mark.parametrize("p", [1, 2, 3])
def test_lp_projector(eps, p, input_data, target_data, perturbation):
    expected_perturbation = torch.clone(perturbation)

    projector = Lp(eps, p)
    projector(perturbation, input=input_data, target=target_data)

    # modify expected_perturbation
    pert_norm = expected_perturbation.norm(p=p)
    if pert_norm > eps:
        expected_perturbation.mul_(eps / pert_norm)

    torch.testing.assert_close(perturbation, expected_perturbation)


def test_mask_projector_repr():
    projector = Mask()
    representation = repr(projector)
    expected_representation = f"{projector.__class__.__name__}()"
    assert representation == expected_representation


def test_mask_projector(input_data, target_data, perturbation):
    expected_perturbation = torch.clone(perturbation)

    projector = Mask()
    projector(perturbation, input=input_data, target=target_data)

    # get expected output
    expected_perturbation.mul_(target_data["perturbable_mask"])

    torch.testing.assert_close(perturbation, expected_perturbation)


def test_compose_repr():
    pass


def test_compose(input_data, target_data):
    pass
