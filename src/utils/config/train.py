import dataclasses as dc

from utils.config.data import DataConfig
from utils.config.feature import FeatureConfig
from utils.config.model import ModelConfig
from utils.config.output import OutputConfig


@dc.dataclass
class ImagePropertiesConfig:
    width: int
    height: int

    def __setattr__(self, key, value):
        match key:
            case "width" | "height":
                if value < 1:
                    raise ValueError(f"{key!r} must be greater than 0")
        super().__setattr__(key, value)

    @property
    def shape(self) -> tuple[int, int]:
        return self.width, self.height


@dc.dataclass
class TrainConfig:
    data: DataConfig
    feature: FeatureConfig
    model: ModelConfig
    out_dirs: OutputConfig
    cv_folds: int
    frequency: str
    image_properties: ImagePropertiesConfig
    seed: int

    __frequency_values = ["full-band", "slow4", "slow5"]

    def __setattr__(self, key, value):
        match key:
            case "frequency":
                if value not in self.__frequency_values:
                    raise ValueError(f"{key!r} must be one of {self.__frequency_values}, got {value!r}")
            case "cv_folds":
                if value < 1:
                    raise ValueError(f"{key!r} must be greater than 0")
        super().__setattr__(key, value)

    def __post_init__(self):
        if "random_state" in self.model.params:
            self.model.params["random_state"] = [self.seed]

    @staticmethod
    def add_type_validation(config_store=None, group=None):
        if config_store is None:
            from utils.config import config_store
        config_store.store(name="_train_type_validation", group=group, node=TrainConfig)
        config_store = DataConfig.add_type_validation()
        config_store = OutputConfig.add_type_validation(config_store)
        config_store = FeatureConfig.add_type_validation(config_store, "feature")
        config_store = ModelConfig.add_type_validation(config_store, "model")

        return config_store