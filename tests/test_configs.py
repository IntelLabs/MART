#
# Copyright (C) 2022 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

from unittest.mock import Mock, patch

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


@patch("torchvision.datasets.imagenet.ImageNet.__init__")
@patch("mart.datamodules.coco.CocoDetection_.__init__")
@patch("torchvision.datasets.CIFAR10.__init__")
def test_experiment_config(
    mock_cifar10: Mock, mock_coco: Mock, mock_imagenet: Mock, cfg_experiment: DictConfig
):
    assert cfg_experiment
    assert cfg_experiment.datamodule
    assert cfg_experiment.model
    assert cfg_experiment.trainer

    # setup mocks
    mock_cifar10.return_value = None
    mock_imagenet.return_value = None
    mock_coco.return_value = None

    HydraConfig().set_config(cfg_experiment)

    hydra.utils.instantiate(cfg_experiment.datamodule)
    hydra.utils.instantiate(cfg_experiment.model)
    hydra.utils.instantiate(cfg_experiment.trainer)
