#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import pytest
import torch

from mart.attack.gradient_modifier import LpNormalizer, Sign


def test_gradient_sign(input_data):
    # Don't share input_data with other tests, because the gradient would be changed.
    input_data = torch.tensor([1.0, 2.0, 3.0])
    input_data.grad = torch.tensor([-1.0, 3.0, 0.0])

    grad_modifier = Sign()
    grad_modifier(input_data)
    expected_grad = torch.tensor([-1.0, 1.0, 0.0])
    torch.testing.assert_close(input_data.grad, expected_grad)


def test_gradient_lp_normalizer():
    # Don't share input_data with other tests, because the gradient would be changed.
    input_data = torch.tensor([1.0, 2.0, 3.0])
    input_data.grad = torch.tensor([-1.0, 3.0, 0.0])

    p = 1
    grad_modifier = LpNormalizer(p)
    grad_modifier(input_data)
    expected_grad = torch.tensor([-0.25, 0.75, 0.0])
    torch.testing.assert_close(input_data.grad, expected_grad)
