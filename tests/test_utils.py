#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import pytest

from mart.utils import FlattenDict


def test_flatten_dict():
    d = {"a": 1, "b": {"c": 2, "d": 3}, "b.e": 4}
    assert FlattenDict(d) == {"a": 1, "b.c": 2, "b.d": 3, "b.e": 4}


def test_flatten_dict_key_collision():
    d = {"a": 1, "b": {"c": 2, "d": 3}, "b.c": 4}
    with pytest.raises(KeyError):
        FlattenDict(d)
