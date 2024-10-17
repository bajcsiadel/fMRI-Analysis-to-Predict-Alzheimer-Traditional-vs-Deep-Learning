import dataclasses as dc
from pathlib import Path

import numpy as np
import hydra
import omegaconf
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torchvision import datasets

from utils.config.model import ModelConfig
from utils.config.resolvers import resolve_results_location
from utils.environment import get_env
from utils.logger import BasicLogger


@dc.dataclass
class Limits:
    train: int = 1000
    test: int = 200


@dc.dataclass
class Config:
    model: ModelConfig
    cifar_location: Path
    cv_folds: int
    image_width: int = 32
    image_height: int = 32
    limit: Limits = dc.field(default_factory=Limits)


def get_data(data_dir: str | Path, train: bool, limit: int = 100_000) -> (np.ndarray, np.ndarray):
    X = []
    y = []

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
    )
    train_data, _ = train_test_split(dataset, train_size=limit, stratify=dataset.targets)

    for image, label in train_data:
        X.append(np.array(image).flatten())
        y.append(label)

    return np.array(X), np.array(y)


def train_model_CIFAR10(
    model: ModelConfig,
    cifar_location: str,
    logger: BasicLogger,
    config: Config,
):
    match model.name:
        case str():
            match model.name.lower():
                case "svm":
                    from sklearn.svm import SVC
                    cls = SVC()
                case "knn":
                    from sklearn.neighbors import KNeighborsClassifier
                    cls = KNeighborsClassifier()
                case _:
                    raise ValueError(f"Model {model} not supported")
        case _:
            raise TypeError(f"Model of type {type(model)} not supported")

    model = GridSearchCV(cls, model.params, cv=StratifiedKFold(n_splits=config.cv_folds))

    # CIFAR10 dataset
    X_train, y_train = get_data(cifar_location, train=True, limit=config.limit.train)
    X_test, y_test = get_data(cifar_location, train=False, limit=config.limit.test)

    model.fit(X_train, y_train)

    cv_results = pd.DataFrame(model.cv_results_)
    logger.info(f"\n{cv_results}")
    cv_results.to_csv(logger.log_dir / "cv_results.csv", index=False)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_pred, y_test)

    # Print the accuracy of the model
    logger.info(f"The model is {accuracy:.2%} accurate")
    logger.info(f"\n{classification_report(y_test, y_pred)}")


@hydra.main(
    version_base=None,
    config_path=get_env("CONFIGURATIONS_LOCATION"),
    config_name="scripts_build_model_CIFAR"
)
def main(cfg: Config):
    logger = BasicLogger(__name__)
    try:
        cfg = omegaconf.OmegaConf.to_object(cfg)

        train_model_CIFAR10(
            model=cfg.model,
            cifar_location=cfg.cifar_location,
            logger=logger,
            config=cfg,
        )
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    config_store = ModelConfig.add_type_validation(group="model")
    config_store.store(name="_script_config_validation", node=Config)

    resolve_results_location()

    main()
