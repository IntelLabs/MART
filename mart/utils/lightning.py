import os
import time
import warnings
from collections import OrderedDict
from glob import glob
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.model_summary import summarize
from omegaconf import DictConfig, OmegaConf

from mart.utils import pylogger, rich_utils

__all__ = [
    "close_loggers",
    "extras",
    "get_metric_value",
    "get_resume_checkpoint",
    "instantiate_callbacks",
    "instantiate_loggers",
    "log_hyperparameters",
    "save_file",
    "task_wrapper",
]

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):

        # apply extra utilities
        extras(cfg)

        # execute the task
        try:
            start_time = time.time()
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = f"'{cfg.task_name}' execution time: {time.time() - start_time} (s)"
            save_file(path, content)  # save task execution time (even if exception occurs)
            close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return wrap


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


@rank_zero_only
def save_file(path, content) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    summary = summarize(model)

    hparams["model/params/total"] = summary.total_parameters
    hparams["model/params/trainable"] = summary.trainable_parameters
    hparams["model/params/non_trainable"] = summary.total_parameters - summary.trainable_parameters

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


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


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


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
