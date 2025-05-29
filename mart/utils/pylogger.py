import logging

from .imports import _HAS_LIGHTNING

if _HAS_LIGHTNING:
    from lightning.pytorch.utilities import rank_zero_only

__all__ = ["get_pylogger"]


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    if _HAS_LIGHTNING:
        logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
        for level in logging_levels:
            setattr(logger, level, rank_zero_only(getattr(logger, level)))
    # Otherwise, fallback to the regular logger if lightning is not installed.

    return logger
