import dataclasses as dc


@dc.dataclass
class ModelConfig:
    name: str
    instance: dict = dc.field(default_factory=dict)
    params: dict = dc.field(default_factory=dict)

    def __setattr__(self, key, value):
        match key:
            case "instance" | "trainer":
                if "_target_" not in value:
                    raise ValueError("Model instance must have a '_target_' attribute")

        super().__setattr__(key, value)

    @staticmethod
    def add_type_validation(config_store=None, group=None):
        if config_store is None:
            from utils.config import config_store
        config_store.store(name="_model_type_validation", group=group, node=ModelConfig)

        return config_store
