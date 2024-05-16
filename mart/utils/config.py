#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import os

import hydra
from hydra import compose as hydra_compose
from hydra import initialize_config_dir
from lightning.pytorch.callbacks.callback import Callback
from omegaconf import OmegaConf

DEFAULT_VERSION_BASE = "1.2"
DEFAULT_CONFIG_DIR = "."
DEFAULT_CONFIG_NAME = "lightning.yaml"

__all__ = ["compose", "instantiate", "Instantiator", "CallbackInstantiator"]


def compose(
    *overrides,
    version_base: str = DEFAULT_VERSION_BASE,
    config_dir: str = DEFAULT_CONFIG_DIR,
    config_name: str = DEFAULT_CONFIG_NAME,
    export_node: str | None = None,
):
    # Add an absolute path {config_dir} to the search path of configs, preceding those in mart.configs.
    if not os.path.isabs(config_dir):
        config_dir = os.path.abspath(config_dir)

    # hydra.initialize_config_dir() requires an absolute path,
    # while hydra.initialize() searches paths relatively to mart.
    with initialize_config_dir(version_base=version_base, config_dir=config_dir):
        cfg = hydra_compose(config_name=config_name, overrides=overrides)

    # Export a sub-tree.
    if export_node is not None:
        for key in export_node.split("."):
            cfg = cfg[key]

    return cfg


def instantiate(cfg_path):
    """Instantiate an object from a Hydra yaml config file."""
    config = OmegaConf.load(cfg_path)
    obj = hydra.utils.instantiate(config)
    return obj


class Instantiator:
    def __new__(cls, cfg_path):
        return instantiate(cfg_path)


class CallbackInstantiator(Callback):
    """Satisfying type checking for Lightning Callback."""

    def __new__(cls, cfg_path):
        obj = instantiate(cfg_path)
        if isinstance(obj, Callback):
            return obj
        else:
            raise ValueError(
                f"We expect to instantiate a lightning Callback from {cfg_path}, but we get {type(obj)} instead."
            )
