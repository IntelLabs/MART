import os

import fire
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def generate(
    *overrides,
    version_base: str = "1.2",
    config_dir: str = ".",
    config_name: str,
    output_node: str = None,
    export_name: str = "output.yaml",
    resolve: bool = False,
):
    # An absolute path {config_dir} is added to the search path of configs, preceding those in mart.configs.
    if not os.path.isabs(config_dir):
        config_dir = os.path.abspath(config_dir)

    with initialize_config_dir(version_base=version_base, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides)

        # Resolve all interpolation.
        if resolve:
            OmegaConf.resolve(cfg)

        # Don't output the whole tree.
        if output_node is not None:
            for key in output_node.split("."):
                cfg = cfg[key]

        OmegaConf.save(config=cfg, f=export_name)
        print(f"Config file saved to {export_name}")


if __name__ == "__main__":
    fire.Fire(generate)
