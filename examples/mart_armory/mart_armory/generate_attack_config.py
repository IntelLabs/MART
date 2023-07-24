# this file acts as a robust starting point for launching hydra runs and multiruns
# can be run from any place

import os
import sys
from pathlib import Path

import hydra
import pyrootutils
from omegaconf import DictConfig, OmegaConf

from mart import utils

log = utils.get_pylogger(__name__)

# project root setup
# uses the current working directory as root.
# sets PROJECT_ROOT environment variable (used in `configs/paths/default.yaml`)
# loads environment variables from ".env" if exists
# adds root dir to the PYTHONPATH (so this file can be run from any place)
# https://github.com/ashleve/pyrootutils
# FIXME: Get rid of pyrootutils if we don't infer config.paths.root from PROJECT_ROOT.
root = Path(os.getcwd())
pyrootutils.set_root(path=root, dotenv=True, pythonpath=True)

config_path = root / "configs"
if not config_path.exists():
    log.warning(f"No config directory found at {config_path}!")
    config_path = "configs"


@hydra.main(version_base="1.2", config_path=config_path, config_name="assemble_attack.yaml")
def main(cfg: DictConfig) -> float:
    if "attack" not in cfg:
        print(
            "Please assemble an attack, e.g., `attack=[object_detection_mask_adversary,data_coco]`"
        )
    else:
        print(OmegaConf.to_yaml(cfg))

    if "output" not in cfg:
        print("You can output config as a yaml file by `output=path/to/file.yaml`")
    else:
        OmegaConf.save(config=cfg, f=cfg.output)
        print(f"Saved config to {cfg.output}")


if __name__ == "__main__":
    ret = main()
    if ret is not None and ret < 0:
        sys.exit(ret)
    else:
        sys.exit(0)
