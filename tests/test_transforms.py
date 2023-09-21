#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import numpy as np
import torch

from mart.transforms.tensor_array import convert


def test_tensor_array_two_way_convert():
    tensor_expected = [{"key": (torch.tensor(1.0), 2)}]
    array_expected = [{"key": (np.array(1.0), 2)}]

    array_result = convert(tensor_expected)
    assert array_expected == array_result

    tensor_result = convert(array_expected)
    assert tensor_expected == tensor_result
