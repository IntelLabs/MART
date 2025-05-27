#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import os

from mart.utils.imports import (
    _HAS_FIFTYONE,
    _HAS_LIGHTNING,
    _HAS_PYCOCOTOOLS,
    _HAS_TIMM,
    _HAS_TORCHVISION,
)


def test_dependency_on_ci():
    if os.getenv("CI") == "true":
        assert (
            _HAS_FIFTYONE
            and _HAS_TIMM
            and _HAS_PYCOCOTOOLS
            and _HAS_TORCHVISION
            and _HAS_LIGHTNING is True
        ), "The dependency is not complete on CI, thus some tests are skipped."
