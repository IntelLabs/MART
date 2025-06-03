import importlib.metadata

from mart import attack as attack
from mart import nn as nn
from mart import optim as optim
from mart import transforms as transforms
from mart import utils as utils
from mart.utils.imports import _HAS_LIGHTNING

if _HAS_LIGHTNING:
    from mart import datamodules as datamodules
    from mart import models as models

__version__ = importlib.metadata.version(__package__ or __name__)
