from ....utils.imports import _HAS_PYCOCOTOOLS, _HAS_TORCHVISION

if _HAS_TORCHVISION and _HAS_PYCOCOTOOLS:
    from .extended import *  # noqa: F403
