from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import OmegaConf


class HydraMartSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Add mart.configs to search path
        search_path.append("hydra-mart", "pkg://mart.configs")


OmegaConf.register_new_resolver("negate", lambda x: -x)
