from importlib import metadata

from mart_armory.attack_wrapper import MartAttack

__version__ = metadata.version(__package__ or __name__)
