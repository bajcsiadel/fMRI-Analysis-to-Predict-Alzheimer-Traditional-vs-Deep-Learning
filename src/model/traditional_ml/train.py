from typing import Callable

import hydra
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from torchvision import transforms

from data import CustomDataset
from utils.config.train import TrainConfig
from utils.logger import TrainLogger


def transform_inputs(image_shape: tuple[int, int]) -> Callable:
    """
    Transforming the input image by resizing and flattening into a 1-dimensional array.

    :param image_shape: new size of the image
    :return: function performing the transform
    """
    resize_transform = transforms.Resize(image_shape)

    def wrapped(input_: np.ndarray):
        output_ = torch.tensor(input_)
        output_ = resize_transform(output_)
        return output_.numpy().flatten()

    return wrapped


def train_model(cfg: TrainConfig, logger: TrainLogger):
    """
    Train a traditional machine learning model.

    :param cfg:
    :param logger:
    """
    model = hydra.utils.instantiate(cfg.model.instance)

    model = GridSearchCV(model, cfg.model.params, cv=cfg.cv_folds)

    train_data = CustomDataset(
        cfg.data.selected_patients,
        cfg.frequency,
        cfg.feature,
        "train",
        transform=transform_inputs(cfg.image_properties.shape)
    )
    test_data = CustomDataset(
        cfg.data.selected_patients,
        cfg.frequency,
        cfg.feature,
        "test",
        transform=transform_inputs(cfg.image_properties.shape)
    )

    # logging information
    logger.info(f"Used feature: {cfg.feature.name}")
    logger.info(model)
    for data in (train_data, test_data):
        logger.info(data)
        logger.debug(data.label_to_target)
        logger.debug(data.metadata.columns)
        logger.debug("\n" + data.metadata.groupby("label")["label"].count().to_string(index=False))
        logger.debug(data.data[0].shape)

    model.fit(train_data.data, train_data.targets)

    logger.info(f"Best parameters: {model.best_params_}")
    logger.info(f"Best score: {model.best_score_}")

    logger.info(f"CV results saved to {logger.log_dir / 'cv_results.csv'}")
    pd.DataFrame(model.cv_results_).to_csv(logger.log_dir / "cv_results.csv", index=False)

    y_pred = model.predict(test_data.data)

    accuracy = accuracy_score(y_pred, test_data.targets)

    # Print the accuracy of the model
    logger.info(f"The model is {accuracy:.2%} accurate")
    logger.info(f"\n{classification_report(test_data.targets, y_pred)}")
