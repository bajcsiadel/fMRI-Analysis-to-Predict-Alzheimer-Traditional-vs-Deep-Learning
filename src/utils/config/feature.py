import dataclasses as dc
from pathlib import Path


@dc.dataclass
class FeatureConfig:
    name: str
    dir_name: str
    root: Path
    extension: str = "npy"
    key: str = "feature"

    @staticmethod
    def add_type_validation(config_store=None, group=None):
        if config_store is None:
            from utils.config import config_store
        config_store.store(name="_feature_type_validation", group=group, node=FeatureConfig)

        return config_store
