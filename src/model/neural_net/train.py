import platform
from typing import Callable

import hydra
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from torchvision import transforms

from model import general
from model.neural_net import SklearnWrapper
from utils.config.train import TrainConfig
from utils.logger import TrainLogger


def get_device(gpu: int) -> str | torch.device:
    """
    Get the device to use for training.

    :param gpu: the GPU to use
    :return: the device
    """
    device = "cpu"
    if gpu >= 0:
        if platform.system() in ["Linux", "Windows"]:
            if torch.cuda.is_available():
                device = "cuda"
                if torch.cuda.device_count() > gpu:
                    device = torch.device(gpu)
        else:
            if torch.backends.mps.is_available():
                device = "mps"

    return device



def transform_inputs(image_shape: tuple[int, int], device: str | torch.device) -> Callable:
    """
    Transforming the input image by resizing and flattening into a 1-dimensional array.

    :param image_shape: new size of the image
    :param device: device to use
    :return: function performing the transform
    """
    resize_transform = transforms.Resize(image_shape)

    def wrapped(input_: np.ndarray):
        output_ = torch.tensor(input_, dtype=torch.float)
        output_ = resize_transform(output_)
        return output_

    return wrapped


def train_model(cfg: TrainConfig, logger: TrainLogger):
    """
    Train a neural network model.

    :param cfg:
    :param logger:
    """
    device = get_device(cfg.gpu)
    train_data, test_data = general.get_data(
        cfg.data.selected_patients, cfg.class_list, cfg.frequency, cfg.feature, logger,
        transform=transform_inputs(cfg.image_properties.shape, device),
        target_transform=lambda target: torch.tensor(target, dtype=torch.float)
    )
    if len(train_data.target_to_label) != 2:
        raise ValueError("Only binary classification is supported")

    n_classes = len(train_data.target_to_label)
    in_channels = train_data.data[0].shape[0]

    cfg.model.params["criterion"] = [hydra.utils.get_class(criterion) for criterion in cfg.model.params["criterion"]]
    cfg.model.params["optimizer"] = [hydra.utils.get_class(optimizer) for optimizer in cfg.model.params["optimizer"]]

    cfg.model.instance["model"]["n_classes"] = n_classes
    cfg.model.instance["model"]["in_channels"] = in_channels
    cfg.model.instance["model"]["device"] = device
    cfg.model.instance["logger"] = logger
    nn_sklearn_model: SklearnWrapper = hydra.utils.instantiate(
        cfg.model.instance,
        device=device,
    )

    model = GridSearchCV(nn_sklearn_model, cfg.model.params, cv=cfg.cv_folds)

    logger.info(model)

    model.fit(train_data.data, train_data.targets.cpu().detach().numpy(), target=train_data.targets, X_valid=test_data.data, y_valid=test_data.targets)

    general.log_cv_results(model, test_data, logger)


def debug_model(cfg: TrainConfig, logger: TrainLogger):
    """
    Test a neural network model on a single batch.

    :param cfg:
    :param logger:
    """
    device = get_device(cfg.gpu)
    train_data, test_data = general.get_data(
        cfg.data.selected_patients, cfg.class_list, cfg.frequency, cfg.feature, logger,
        transform=transform_inputs(cfg.image_properties.shape, device),
        target_transform=lambda target: torch.tensor(target, dtype=torch.float)
    )
    if len(train_data.target_to_label) != 2:
        raise ValueError("Only binary classification is supported")

    n_classes = len(train_data.target_to_label)
    in_channels = train_data.data[0].shape[0]

    cfg.model.params["criterion"] = [hydra.utils.get_class(criterion) for criterion in cfg.model.params["criterion"]]
    cfg.model.params["optimizer"] = [hydra.utils.get_class(optimizer) for optimizer in cfg.model.params["optimizer"]]

    multi_option_params = [len(param) > 1 for param in cfg.model.params.values()]
    if any(multi_option_params):
        logger.warning("Debugging does not perform GridSearchCV, only the first option will be used.")
    cfg.model.params = {key: value[0] for key, value in cfg.model.params.items()}

    logger.info(cfg.model.params)

    cfg.model.instance["model"]["n_classes"] = n_classes
    cfg.model.instance["model"]["in_channels"] = in_channels
    cfg.model.instance["model"]["device"] = device
    cfg.model.instance["logger"] = logger
    model: SklearnWrapper = hydra.utils.instantiate(
        cfg.model.instance,
        device=device,
        **cfg.model.params,
    )

    logger.info(model)

    X_train, y_train = train_data.get_batch(cfg.model.params["batch_size"], 0)

    logger.debug(y_train.unique(return_counts=True))

    X_valid, y_valid = train_data.get_batch(cfg.model.params["batch_size"], 1)

    logger.debug(y_valid.unique(return_counts=True))

    model.fit(X_train, y_train.cpu().detach().numpy(), target=y_train, X_valid=X_valid, y_valid=y_valid)

    general.log_results(model, test_data, logger)
