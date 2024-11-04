import copy
from typing import Callable

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data import CustomDataset
from utils.config.data import CSVFileConfig
from utils.config.feature import FeatureConfig
from utils.logger import TrainLogger


def get_data(
    meta_file: CSVFileConfig,
    frequency: str,
    feature: FeatureConfig,
    logger: TrainLogger,
    transform: Callable = None,
    target_transform: Callable = None
) -> tuple[CustomDataset, CustomDataset]:
    """
    Get the training and testing data.

    :param meta_file: metadata file information
    :param frequency: frequency to use in the filter
    :param feature: feature information
    :param logger:
    :param transform: transformation to apply to the data
    :return: training and testing data
    """
    train_data = CustomDataset(
        meta_file,
        frequency,
        feature,
        "train",
        transform=transform,
        target_transform=target_transform,
    )
    test_data = CustomDataset(
        meta_file,
        frequency,
        feature,
        "test",
        transform=transform,
        target_transform=target_transform,
    )

    for data in (train_data, test_data):
        logger.info(data)
        logger.debug(f"Class target mapping: {data.label_to_target}")
        logger.debug(f"Columns: {data.metadata.columns}")
        logger.debug(f"Input shape: {data.data[0].shape}")
        logger.debug(f"Input size: {len(data.data)}")
        logger.debug(f"Target size: {len(data.targets)}")

        logger.debug("Label distribution:")
        label_count = data.metadata.groupby("label")["label"].count().to_frame()
        label_count["count"] = label_count["label"]
        del label_count["label"]
        label_count["ratio"] = label_count["count"] / label_count["count"].sum()
        label_count.reset_index(inplace=True)
        logger.debug(label_count.to_string(index=False))
        del label_count

    return train_data, test_data


def log_cv_results(model, test_data: CustomDataset, logger: TrainLogger):
    """
    Log the results of the model.
    :param model: trained model
    :param test_data: testing data
    :param logger:
    """
    logger.info(f"Best parameters: {model.best_params_}")
    logger.info(f"Best score: {model.best_score_}")
    logger.info(f"CV results saved to {logger.log_dir / 'cv_results.csv'}")
    pd.DataFrame(model.cv_results_).to_csv(logger.log_dir / "cv_results.csv",
                                           index=False)
    log_results(model, test_data, logger)



def log_results(model, test_data: CustomDataset, logger: TrainLogger):
    """
    Log the results of the model.
    :param model: trained model
    :param test_data: testing data
    :param logger:
    """
    y_pred = model.predict(test_data.data)
    y_pred_proba = model.predict_proba(test_data.data)

    if type(y_pred) == tuple:
        logger.info(f"The model loss on the test set: {y_pred[1]} {y_pred_proba[1]}")
        y_pred = y_pred[0]
        y_pred_proba = y_pred_proba[0]

    accuracy = accuracy_score(y_pred, test_data.targets)
    predictions = copy.deepcopy(test_data.metadata)
    predictions["prediction"] = [test_data.target_to_label[pred] for pred in y_pred]
    for target, label in test_data.target_to_label.items():
        predictions[f"probability_{label}"] = y_pred_proba[:, target]
    predictions.to_csv(logger.log_dir / "predictions.csv", index=False)
    confusion = confusion_matrix(test_data.targets, y_pred)
    classes = [test_data.target_to_label[i] for i in range(confusion.shape[0])]
    pd.DataFrame(confusion, index=pd.Index(classes, name="Actual"),
                 columns=pd.Index(classes, name="Predicted")).to_csv(
        logger.log_dir / "confusion_matrix.csv"
    )
    # Print the accuracy of the model
    logger.info(f"The model is {accuracy:.2%} ({accuracy}) accurate")
    logger.info(f"\n{classification_report(test_data.targets, y_pred)}")
