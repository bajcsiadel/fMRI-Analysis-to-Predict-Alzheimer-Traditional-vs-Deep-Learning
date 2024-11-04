from typing import Callable

import hydra
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from torchvision import transforms

from model import general
from utils.config.train import TrainConfig
from utils.logger import TrainLogger

from src.model.general import log_cv_results


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

    train_data, test_data = general.get_data(
        cfg.data.selected_patients, cfg.frequency, cfg.feature, logger,
        transform=transform_inputs(cfg.image_properties.shape),
    )

    # logging information
    logger.info(f"Used feature: {cfg.feature.name}")
    logger.info(model)

    model.fit(train_data.data, train_data.targets)

    log_cv_results(model, test_data, logger)
