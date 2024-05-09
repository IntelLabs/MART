#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from importlib.util import find_spec


def has(module_name):
    return find_spec(module_name) is not None


_HAS_FIFTYONE = has("fiftyone")
_HAS_TORCHVISION = has("torchvision")
_HAS_TIMM = has("timm")
