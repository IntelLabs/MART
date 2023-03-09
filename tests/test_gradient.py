#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import pytest
import torch

from mart.attack.gradient_modifier import LpNormalizer, Sign


def test_gradient_sign(input_data):
    gradient = Sign()
    output = gradient(input_data)
    expected_output = input_data.sign()
    torch.testing.assert_close(output, expected_output)


def test_gradient_lp_normalizer(input_data):
    p = 1
    gradient = LpNormalizer(p)
    output = gradient(input_data)
    expected_output = input_data / input_data.norm(p=p)
    torch.testing.assert_close(output, expected_output)
