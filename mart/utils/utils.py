import os
import warnings
from glob import glob
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from . import pylogger, rich_utils

__all__ = [
    "extras",
    "get_metric_value",
    "get_resume_checkpoint",
    "task_wrapper",
    "flatten_dict",
]

log = pylogger.get_pylogger(__name__)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:

        ...

        return metric_dict, object_dict
    ```
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def get_resume_checkpoint(config: DictConfig) -> Tuple[DictConfig]:
    """Resume a task from an existing checkpoint along with the config."""

    resume_checkpoint = None

    if config.get("resume"):
        resume_filename = os.path.join("checkpoints", "*.ckpt")
        resume_dir = hydra.utils.to_absolute_path(config.resume)

        # If we pass an explicit path, parse it to get base directory and checkpoint filename
        if os.path.isfile(resume_dir):
            resume_path = Path(resume_dir)

            # Resume path looks something like:
            #   /path/to/checkpoints/checkpoint_name.ckpt
            #  So we pass in to base directory and checkpoint filename and "checkpoints" directory
            resume_dir = os.path.join(*resume_path.parts[:-2])
            resume_filename = os.path.join(*resume_path.parts[-2:])

        # Get old and new overrides and combine them
        current_overrides = HydraConfig.get().overrides.task
        overrides_config = OmegaConf.load(os.path.join(resume_dir, ".hydra", "overrides.yaml"))
        overrides = overrides_config + current_overrides

        # Find checkpoint and set PL trainer to resume
        resume_checkpoint = glob(os.path.join(resume_dir, resume_filename))

        if len(resume_checkpoint) == 0:
            msg = f"No checkpoint found in {os.path.join(resume_dir, resume_filename)}!"
            log.error(msg)
            raise Exception(msg)

        # If we find more than 1 checkpoint, tell the user to be more explicit about their choice
        # of checkpoint to resume from!
        if len(resume_checkpoint) > 1:
            msg = f"Found more than 1 checkpoint in {resume_dir} so you must pass a checkpoint path to resume:"
            log.error(msg)
            for path in resume_checkpoint:
                log.error(f"  {path}")
            raise Exception(msg)

        resume_checkpoint = resume_checkpoint[0]
        log.info(f"Resuming from {resume_checkpoint}")
        # Save the ckpt_path in cfg for fit() and test().
        # This override won't be written to disk .hydra/overrides.yaml
        overrides += [f"+ckpt_path={resume_checkpoint}"]

        # Load hydra.conf and use job config name to load original config with overrides
        hydra_config = OmegaConf.load(os.path.join(resume_dir, ".hydra", "hydra.yaml"))
        config_name = hydra_config.hydra.job.config_name
        config = hydra.compose(config_name, overrides=overrides)

    return config


def flatten_dict(d, delimiter="."):
    def get_dottedpath_items(d: dict, parent: Optional[str] = None):
        """Get pairs of the dotted path and the value from a nested dictionary."""
        for name, value in d.items():
            path = f"{parent}{delimiter}{name}" if parent else name
            if isinstance(value, dict):
                yield from get_dottedpath_items(value, parent=path)
            else:
                yield path, value

    ret = {}
    for key, value in get_dottedpath_items(d):
        if key in ret:
            raise KeyError(f"Key collision when flattening a dictionary: {key}")
        ret[key] = value

    return ret
