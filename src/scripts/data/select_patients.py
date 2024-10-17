import dataclasses as dc
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
        logger.info(data_distribution(metadata, "label").to_string(index=False))

        logger.info("Average age per group:")
        logger.info(metadata.groupby("label")["age"].mean())

        cn = metadata[metadata["label"] == "CN"]
        ad = metadata[metadata["label"] == "AD"]

        logger.info("Balancing the groups by age:")
        selected_cn = None
        missing = {}
        for (age_value, rows) in ad.groupby("age"):
            cn_with_age = cn[cn["age"] == age_value]
            if len(cn_with_age) < len(rows):
                current_selection = cn_with_age.values
                missing[age_value] = len(rows) - len(cn_with_age)
            else:
                current_selection = cn[cn["age"] == age_value].sample(
                    n=len(rows)
                ).values

            if selected_cn is None:
                selected_cn = current_selection
            else:
                selected_cn = np.vstack((selected_cn, current_selection))

        used = []
        for age_value in missing.keys():
            age_diff = np.abs(cn["age"].values - age_value)
            for _ in range(missing[age_value]):
                closest = np.argmin(age_diff)
                while closest in used:
                    age_diff[closest] = 100
                    closest = np.argmin(age_diff)

                selected_cn = np.vstack((selected_cn, cn.iloc[closest].values))
                used.append(closest)
                age_diff[closest] = 100
        cn = pd.DataFrame(selected_cn, columns=cn.columns)

        data = pd.concat([cn, ad], ignore_index=True)

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
