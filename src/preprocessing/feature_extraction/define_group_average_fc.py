from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr

from preprocessing.feature_extraction.features.high_order.associated import \
    static_associated_high_order_fc
from preprocessing.feature_extraction.features.high_order.topographical import \
    static_topographical_high_order_fc
from preprocessing.feature_extraction.features.low_order import static_low_order_fc
from utils.environment import get_env


FREQUENCY = "full-band"
ADNI_ROOT = Path(get_env("DATA_ROOT"), "ADNI")
SIGNAL_LOCATION = ADNI_ROOT / "Signals" / f"{FREQUENCY}_ROISignals_FunImgARCWSF"
SIGNAL_FILE_TEMPLATE = "ROISignals_{}.txt"

metadata = pd.read_csv(ADNI_ROOT / "metadata.csv", header=0, index_col=0)

Cs = {}
Hs = {}
As = {}
for label in metadata["label"].unique():
    patient_ids = metadata[metadata["label"] == label].index

    Cs[label] = []
    Hs[label] = []
    As[label] = []

    for patient_id in patient_ids:
        try:
            signals = np.loadtxt(SIGNAL_LOCATION / SIGNAL_FILE_TEMPLATE.format(patient_id))
        except FileNotFoundError:
            print(f"{patient_id} not found")
            continue

        C = static_low_order_fc(signals)
        H = static_topographical_high_order_fc(C=C)
        A = static_associated_high_order_fc(C=C, H=H)

        Cs[label].append(C)
        Hs[label].append(H)
        As[label].append(A)

    Cs[label] = np.array(Cs[label]).squeeze()
    Hs[label] = np.array(Hs[label]).squeeze()
    As[label] = np.array(As[label]).squeeze()

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for i, (arr, title) in enumerate([
        (Cs[label], "Low-order"),
        (Hs[label], "Topographical high-order"),
        (As[label], "Associated high-order")
    ]):
        sns.heatmap(
            arr.mean(0),
            cmap="RdYlBu_r",
            ax=ax[i],
            vmin=-0.6,
            vmax=1.0,
        )
        ax[i].axis("off")
        ax[i].set_title(title)
    plt.suptitle(f"Group average for {label}")
    plt.show()

class_combinations = list(combinations(Cs.keys(), 2))
for arr, feature_type in [
    (Cs, "Low-order"),
    (Hs, "Topographical high-order"),
    (As, "Associated high-order")
]:
    for class_combination in class_combinations:
        params = []
        min_length = np.inf
        for label in class_combination:
            params.append(arr[label])
            if len(arr[label]) < min_length:
                min_length = len(arr[label])

        r2 = np.array([
            [
                pointbiserialr((params[0][:, i, j])[:min_length], (params[1][:, i, j])[:min_length]).correlation ** 2
                for i in range(params[0].shape[1])
            ]
            for j in range(params[0].shape[2])
        ])
        r2 = np.triu(r2)
        fig, ax = plt.subplots()
        sns.heatmap(r2, cmap="Reds", ax=ax)
        plt.title(f"Separation between {class_combination} for {feature_type}")
        plt.show()
