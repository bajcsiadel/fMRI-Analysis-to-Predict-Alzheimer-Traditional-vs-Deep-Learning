import dataclasses as dc
from ast import Index

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from utils.config.resolvers import resolve_results_location
from utils.config.types import DataConfig
from utils.environment import get_env
from utils.logger import BasicLogger


@dc.dataclass
class Config:
    data: DataConfig
    seed: int
    test_size: float


def data_distribution(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Get the distribution of a column in a DataFrame.

    :param data: metadata
    :param column_name: name of the column
    :return: the distribution of values in the column
    """
    result = pd.DataFrame(columns=["count", "percentage"])
    result["count"] = data[column_name].value_counts()
    result["percentage"] = result["count"] / result["count"].sum()
    result.reset_index(inplace=True)
    return result


@hydra.main(
    version_base=None,
    config_path=get_env("CONFIGURATIONS_LOCATION"),
    config_name="scripts_data_select_patients"
)
def main(cfg: Config):
    logger = BasicLogger(__name__)

    try:
        cfg = OmegaConf.to_object(cfg)

        metadata = pd.read_csv(
            cfg.data.metadata.filename,
            **cfg.data.metadata.parameters
        )
        logger.info(f"Data read from {cfg.data.metadata.filename}")

        logger.info(f"Number of entries: {len(metadata)}")
        logger.info(
            f"Average session per patient: "
            f"{metadata.groupby('patient_id').size().mean()}"
        )

        logger.info("Sex distribution:")
        logger.info(data_distribution(metadata, "sex").to_string(index=False))

        logger.info("Label distribution:")
        label_count = data_distribution(metadata, "label")
        logger.info(data_distribution(metadata, "label").to_string(index=False))

        logger.info("Average age per group:")
        logger.info(metadata.groupby("label")["age"].mean())

        ad_count = label_count[label_count["label"] == "AD"]["count"].values[0]
        label_count = label_count[label_count["count"] >= ad_count]
        logger.info("Labels to keep:")
        logger.info(label_count.to_string(index=False))

        ad = metadata[metadata["label"] == "AD"]
        label_data = {label: metadata[metadata["label"] == label] for label in label_count["label"] if label != "AD"}

        logger.info("Balancing the groups by age:")
        selected = {label: None for label in label_count["label"] if label != "AD"}
        missing = {label: {} for label in label_count["label"] if label != "AD"}
        for (age_value, rows) in ad.groupby("age"):
            for other_label, other_label_data in label_data.items():
                with_age = other_label_data[other_label_data["age"] == age_value]
                if len(with_age) < len(rows):
                    current_selection = with_age.values
                    missing[other_label][age_value] = len(rows) - len(with_age)
                else:
                    current_selection = with_age.sample(n=len(rows)).values

                if selected[other_label] is None:
                    selected[other_label] = current_selection
                else:
                    selected[other_label] = np.vstack((selected[other_label], current_selection))

        for other_label in missing.keys():
            other_label_data = label_data[other_label]
            used_patients = selected[other_label][:, 0].tolist()
            patient_ids = other_label_data["filename"].values.tolist()
            for age_value in missing[other_label].keys():
                age_diff = np.abs(other_label_data["age"].values - age_value)
                age_diff[age_diff == 0] = 100  # patients with the given age are already selected
                for _ in range(missing[other_label][age_value]):
                    closest = np.argmin(age_diff)
                    while (patient_id := patient_ids[closest]) in used_patients:
                        age_diff[closest] = 100
                        closest = np.argmin(age_diff)

                    selected[other_label] = np.vstack((selected[other_label], other_label_data.iloc[closest].values))
                    used_patients.append(patient_id)
                    age_diff[closest] = 100

        final_data = {label: pd.DataFrame(selected_data, columns=ad.columns) for label, selected_data in selected.items()}
        final_data["ad"] = ad

        data = pd.concat(final_data, ignore_index=True)

        logger.info("Label distribution after balancing:")
        logger.info(data_distribution(data, "label").to_string(index=False))

        logger.info("Average age per group after balancing:")
        logger.info(data.groupby("label")["age"].mean())

        train_data, test_data = train_test_split(
            data,
            test_size=cfg.test_size,
            random_state=cfg.seed,
            shuffle=True,
            stratify=data["label"],
        )
        train_data["set"] = ["train"] * len(train_data)
        test_data["set"] = ["test"] * len(test_data)

        data = pd.concat((train_data, test_data), ignore_index=True)

        logger.info("Set distribution in train set:")
        logger.info(data_distribution(data, "set").to_string(index=False))

        logger.info("Label distribution in train set:")
        logger.info(
            data_distribution(
                train_data, "label"
            ).to_string(index=False)
        )

        logger.info("Average age per group in train:")
        logger.info(train_data.groupby("label")["age"].mean())

        logger.info("Label distribution in test set:")
        logger.info(
            data_distribution(
                test_data, "label"
            ).to_string(index=False)
        )

        logger.info("Average age per group in test:")
        logger.info(test_data.groupby("label")["age"].mean())

        data.to_csv(
            cfg.data.metadata.filename.with_name(
                cfg.data.selected_patients.filename.name
            ),
            index=False
        )
    except Exception as e:
        logger.exception(e)


config_store = DataConfig.add_type_validation()
config_store.store(name="_script_config_validation", node=Config)

resolve_results_location()

main()
