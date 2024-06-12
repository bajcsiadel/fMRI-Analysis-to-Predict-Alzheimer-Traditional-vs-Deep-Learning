import dataclasses as dc


@dc.dataclass
class OutputConfig:
    metadata_dir: str
    checkpoints_dir: str

    @staticmethod
    def add_type_validation(config_store=None, group=None):
        if config_store is None:
            from utils.config import config_store
        config_store.store(name="_output_type_validation", group=group, node=OutputConfig)

        return config_store
