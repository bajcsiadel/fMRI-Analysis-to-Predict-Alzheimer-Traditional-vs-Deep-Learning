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
    classes: str
    data: DataConfig
    feature: FeatureConfig
    model: ModelConfig
    out_dirs: OutputConfig
    cv_folds: int
    frequency: str
    gpu: int
    image_properties: ImagePropertiesConfig
    seed: int

    __class_list = []

    __classes_values = ["AD", "CN", "EMCI", "LMCI"]
    __frequency_values = ["full-band", "slow4", "slow5"]

    def __setattr__(self, key, value):
        match key:
            case "frequency":
                if value not in self.__frequency_values:
                    raise ValueError(f"{key!r} must be one of {self.__frequency_values}, got {value!r}")
            case "cv_folds":
                if value < 1:
                    raise ValueError(f"{key!r} must be greater than 0")
            case "classes":
                self.__class_list = value.split("-vs-")
                if any(class_ not in self.__classes_values for class_ in self.__class_list):
                    raise ValueError(f"{key!r} must be a combination of {self.__classes_values}, got {value!r}")
        super().__setattr__(key, value)

    def __post_init__(self):
        if "random_state" in self.model.params:
            self.model.params["random_state"] = [self.seed]

    @property
    def class_list(self) -> list[str]:
        return self.__class_list.copy()

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