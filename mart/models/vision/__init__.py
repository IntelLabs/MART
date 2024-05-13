#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from ...utils.imports import _HAS_TORCHVISION

if _HAS_TORCHVISION:
    from .dual_mode import *  # noqa: F403
