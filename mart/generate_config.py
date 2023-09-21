#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import os

import fire
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def generate(
    *overrides,
    version_base: str = "1.2",
    config_dir: str = "configs",
    config_name: str = "lightning.yaml",
    export_node: str = None,
    export_name: str = "output.yaml",
    resolve: bool = False,
):
    # An absolute path {config_dir} is added to the search path of configs, preceding those in mart.configs.
    if not os.path.isabs(config_dir):
        config_dir = os.path.abspath(config_dir)

    with initialize_config_dir(version_base=version_base, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides)

        # Export a sub-tree.
        if export_node is not None:
            for key in export_node.split("."):
                cfg = cfg[key]

        # Resolve all interpolation in the sub-tree.
        if resolve:
            OmegaConf.resolve(cfg)

        # Create folders for output if necessary.
        folder = os.path.dirname(export_name)
        if folder != "" and not os.path.isdir(folder):
            os.makedirs(folder)

        OmegaConf.save(config=cfg, f=export_name)
        print(f"Config file saved to {export_name}")


if __name__ == "__main__":
    fire.Fire(generate)
