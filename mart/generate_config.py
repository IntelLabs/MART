#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import os

import fire
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def mart_compose(
    *overrides,
    version_base: str = "1.2",
    config_dir: str = "configs",
    config_name: str = "lightning.yaml",
    export_node: str = None,
):
    # Add an absolute path {config_dir} to the search path of configs, preceding those in mart.configs.
    if not os.path.isabs(config_dir):
        config_dir = os.path.abspath(config_dir)

    # initialize_config_dir() requires an absolute path.
    with initialize_config_dir(version_base=version_base, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides)

    # Export a sub-tree.
    if export_node is not None:
        for key in export_node.split("."):
            cfg = cfg[key]

    return cfg


def get_yaml_cfg(cfg, resolve: bool = False):
    # Resolve all interpolation in the sub-tree.
    if resolve:
        OmegaConf.resolve(cfg)

    return OmegaConf.to_yaml(cfg)


def main(
    *overrides,
    version_base: str = "1.2",
    config_dir: str = "configs",
    config_name: str = "lightning.yaml",
    export_node: str = None,
    resolve: bool = False,
):
    cfg = mart_compose(
        *overrides,
        version_base=version_base,
        config_dir=config_dir,
        config_name=config_name,
        export_node=export_node,
    )

    cfg_yaml = get_yaml_cfg(cfg, resolve=resolve)

    # OmegaConf.to_yaml() already ends with `\n`.
    print(cfg_yaml, end="")


if __name__ == "__main__":
    fire.Fire(main)
