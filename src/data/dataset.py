import copy
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.config.data import CSVFileConfig
from utils.config.feature import FeatureConfig


def _identity_transform(x):
    return x


class CustomDataset(Dataset):
    def __init__(self, metafile: CSVFileConfig, classes: list[str], frequency: str, feature: FeatureConfig, set_: str, transform: Callable = None, target_transform: Callable = None):
        self._data_details = metafile
        self._feature_details = feature

        self._metadata = pd.read_csv(metafile.filename, **metafile.parameters)

        self._metadata = self._metadata[self._metadata["label"].isin(classes)]

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

        if target_transform is None:
            self._target_transform = _identity_transform
        else:
            self._target_transform = target_transform

        self._targets = (self._metadata["label"]
                         .map(lambda label: self._label_to_target[label])
                         .map(self._target_transform).values)
        self._data = [
            self._transform(
                np.load(
                    feature.root / frequency / feature.dir_name / f"{row['filename']}.{feature.extension}"
                )[feature.key]
            )
            for _, row in self._metadata.iterrows()
        ]

        match self._data[0]:
            case np.ndarray():
                self._data = np.array(self._data)
            case torch.Tensor():
                self._data = torch.stack(self._data)
                self._targets = torch.stack(self._targets.tolist())

        self.batches = None

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
        return copy.deepcopy(self._targets)

    @property
    def data(self):
        return copy.deepcopy(self._data)

    @property
    def n_classes(self):
        return len(np.unique(self._targets))

    @property
    def feature_name(self):
        return self._feature_details.name

    def get_batch(self, batch_size: int, index: int):
        if self.batches is None:
            self.batches = []
            for class_index in range(self.n_classes):
                indices = np.where(self._targets == class_index)[0]
                samples_in_batch = int(len(indices) / len(self) * batch_size)
                np.random.shuffle(indices)
                if len(self.batches) == 0:
                    self.batches = [
                        indices[i:i+samples_in_batch]
                        for i in range(0, len(indices), samples_in_batch)
                    ]
                else:
                    self.batches = [
                        np.concatenate((self.batches[batch_index], indices[i:i+samples_in_batch]))
                        for batch_index, i in enumerate(range(0, len(indices), samples_in_batch))
                    ]
        indices = self.batches[index]
        return self._data[indices], self._targets[indices]

    def __repr__(self):
        return f"CustomDataset(file={self._data_details.filename}, set={self._set}, n_samples={len(self)})"

    def __len__(self):
        return len(self._metadata)

    def __getitem__(self, idx):
        return self._data[idx], self._targets[idx]
