#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import os
from pathlib import Path

import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

root = Path(os.getcwd())
pyrootutils.set_root(path=root, dotenv=True, pythonpath=True)

experiments_names = [
    "ArmoryCarlaOverObjDet_TorchvisionFasterRCNN",
]


# Loads the configuration file from a given experiment
def get_cfg(experiment):
    with initialize(version_base="1.2", config_path="../configs"):
        params = "experiment=" + experiment
        cfg = compose(config_name="lightning.yaml", return_hydra_config=True, overrides=[params])
    return cfg


@pytest.fixture(scope="function", params=experiments_names)
def cfg_experiment(request) -> DictConfig:
    cfg = get_cfg(request.param)

    yield cfg

    GlobalHydra.instance().clear()
