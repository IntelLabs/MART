#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import os
from pathlib import Path

import pyrootutils
import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

root = Path(os.getcwd())
pyrootutils.set_root(path=root, dotenv=True, pythonpath=True)

experiments_names = [
    "CIFAR10_CNN",
    "CIFAR10_CNN_Adv",
    "ImageNet_Timm",
    "COCO_TorchvisionFasterRCNN",
    "COCO_TorchvisionFasterRCNN_Adv",
    "COCO_TorchvisionRetinaNet",
]


# Loads the configuration file from a given experiment
def get_cfg(experiment):
    with initialize(version_base="1.2", config_path="../mart/configs"):
        params = "experiment=" + experiment
        cfg = compose(config_name="lightning.yaml", return_hydra_config=True, overrides=[params])
    return cfg


@pytest.fixture(scope="function", params=experiments_names)
def cfg_experiment(request) -> DictConfig:
    cfg = get_cfg(request.param)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def input_data():
    image_size = (3, 32, 32)
    return torch.randint(0, 256, image_size, dtype=torch.float)


@pytest.fixture(scope="function")
def target_data():
    image_size = (3, 32, 32)
    return {"perturbable_mask": torch.ones(*image_size), "file_name": "test.jpg"}


@pytest.fixture(scope="function")
def perturbation():
    torch.manual_seed(0)
    perturbation = torch.randint(0, 256, (3, 32, 32), dtype=torch.float)
    return perturbation
