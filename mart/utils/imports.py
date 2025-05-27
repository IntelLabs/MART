#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from importlib.util import find_spec

from .pylogger import get_pylogger

logger = get_pylogger(__name__)


def has(module_name):
    module = find_spec(module_name)
    if module is None:
        logger.warn(f"{module_name} is not installed, so some features in MART are unavailable.")
        return False
    else:
        return True


# Do not forget to add dependency checks on CI in `tests/test_dependency.py`
_HAS_FIFTYONE = has("fiftyone")
_HAS_TORCHVISION = has("torchvision")
_HAS_TIMM = has("timm")
_HAS_PYCOCOTOOLS = has("pycocotools")
_HAS_LIGHTNING = has("lightning")
