from typing import Callable

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.config.data import CSVFileConfig
from utils.config.feature import FeatureConfig


def _identity_transform(x):
    return x


class CustomDataset(Dataset):
    def __init__(self, metafile: CSVFileConfig, frequency: str, feature: FeatureConfig, set_: str, transform: Callable = None):
        self._data_details = metafile
        self._feature_details = feature

        self._metadata = pd.read_csv(metafile.filename, **metafile.parameters)

        if set_ not in ["all", "train", "test"]:
            raise ValueError(
                f"set_ must be one of ['all', 'train', 'test'], got {set_}"
            )
        if set_ != "all":
            self._metadata = self._metadata[self._metadata["set"] == set_]
        self._set = set_

        if frequency not in ["full-band", "slow4", "slow5"]:
            raise ValueError(
                f"frequency must be one of ['full-band', 'slow4', 'slow5'], got {frequency}"
            )
        self._frequency = frequency

        self._label_to_target = {
            class_name: i
            for i, class_name in enumerate(np.unique(self._metadata["label"].values))
        }

        if transform is None:
            self._transform = _identity_transform
        else:
            self._transform = transform

        self._targets = self._metadata["label"].map(self._target_transform).values
        self._data = np.array(
            [
                self._transform(
                    np.load(
                        feature.root / frequency / feature.dir_name / f"{row['filename']}.{feature.extension}"
                    )[feature.key]
                )
                for _, row in self._metadata.iterrows()
            ]
        )

    @property
    def metadata(self):
        return self._metadata.copy()

    @property
    def label_to_target(self):
        return self._label_to_target.copy()

    @property
    def target_to_label(self):
        return {v: k for k, v in self._label_to_target.items()}

    @property
    def targets(self):
        return self._targets.copy()

    @property
    def data(self):
        return self._data.copy()

    def _target_transform(self, label):
        return self._label_to_target[label]

    def __repr__(self):
        return f"CustomDataset(file={self._data_details.filename}, set={self._set}, n_samples={len(self)})"

    def __len__(self):
        return len(self._metadata)

    def __getitem__(self, idx):
        return self._data[idx], self._targets[idx]
