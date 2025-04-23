# this file acts as a robust starting point for launching hydra runs and multiruns
# can be run from any place

import os
import sys

import hydra
import pyrootutils
from omegaconf import DictConfig

from mart import utils

log = utils.get_pylogger(__name__)

# project root setup
# uses the current working directory as root.
# sets PROJECT_ROOT environment variable (used in `configs/paths/default.yaml`)
# loads environment variables from ".env" if exists
# adds root dir to the PYTHONPATH (so this file can be run from any place)
# https://github.com/ashleve/pyrootutils
# FIXME: Get rid of pyrootutils if we don't infer config.paths.root from PROJECT_ROOT.
# Hydra does not support Posix path after 1.2.0: https://github.com/facebookresearch/hydra/commit/53d07f56a272485cc81596d23aad33e18e007091
# Use string path instead.
root = os.getcwd()
pyrootutils.set_root(path=root, dotenv=True, pythonpath=True)

config_path = os.path.join(root, "configs")
if not config_path.exists():
    log.warning(f"No config directory found at {config_path}!")
    config_path = "configs"


@hydra.main(version_base="1.2", config_path=config_path, config_name="lightning.yaml")
def main(cfg: DictConfig) -> float:

    if cfg.resume is None and ("datamodule" not in cfg or "model" not in cfg):
        log.fatal("")
        log.fatal("Please specify an experiment to run, e.g.")
        log.fatal(
            "$ python -m mart experiment=CIFAR10_CNN fit=false +trainer.limit_test_batches=1"
        )
        log.fatal("or specify a checkpoint to resume, e.g.")
        log.fatal("$ python -m mart resume=logs/my_task_name/checkpoints/last.ckpt")
        log.fatal("")
        return -1

    # imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from mart.tasks.lightning import lightning
    from mart.utils import get_metric_value, get_resume_checkpoint

    # Resume and modify configs at the earliest point.
    # The actual checkpoint path is in cfg.ckpt_path
    cfg = get_resume_checkpoint(cfg)

    # train the model
    metric_dict, _ = lightning(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    ret = main()
    if ret is not None and ret < 0:
        sys.exit(ret)
    else:
        sys.exit(0)
