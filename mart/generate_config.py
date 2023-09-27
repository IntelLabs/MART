#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import fire

from .utils.config import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_NAME,
    DEFAULT_VERSION_BASE,
    compose,
    get_yaml_cfg,
)


def main(
    *overrides,
    version_base: str = DEFAULT_VERSION_BASE,
    config_dir: str = DEFAULT_CONFIG_DIR,
    config_name: str = DEFAULT_CONFIG_NAME,
    export_node: str | None = None,
    resolve: bool = False,
):
    cfg = compose(
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
