import importlib

from mart import attack as attack
from mart import data as data
from mart import models as models
from mart import nn as nn
from mart import optim as optim
from mart import transforms as transforms
from mart import utils as utils

__version__ = importlib.metadata.version(__package__ or __name__)
