import numpy as np
from icecream import ic

from preprocessing.feature_extraction.helpers import apply_sliding_window, correlation
from utils.environment import get_env


def dynamic_low_order_fc(
    signal: np.ndarray, window_length: int, stride: int = 1
) -> np.ndarray:
    """
    Function to represent a low order feature extractor presented in [Zhang-2017]_.
    After the sub-series are defined a correlation matrix is constructed using
    Pearson's correlation. In the original article the following values were used:
    window_length = 70 and stride = 1.

    .. [Zhang-2017] Zhang, Y., Zhang, H., Chen, X., Lee, S.-W., & Shen, D. (2017).
        Hybrid High-order Functional Connectivity Networks Using Resting-state
        Functional MRI for Mild Cognitive Impairment Diagnosis. Scientific Reports,
        7(1), 6530. https://doi.org/10.1038/s41598-017-06509-0

    :param signal: The extracted ROI signals
    :param window_length: The length of the sliding window.
    :param stride: The stride used for the sliding window. Defaults to ``1``.
    :return: The computed dynamic low-order FC
    """
    X = apply_sliding_window(signal, window_length=window_length, stride=stride)
    # defining correlation
    C = np.empty((X.shape[0], X.shape[2], X.shape[2]))
    for k in range(X.shape[0]):
        # X is centralized and normalized while computing the correlation
        C[k] = correlation(X[k])

    return C


def static_low_order_fc(signal: np.ndarray) -> np.ndarray:
    """
    Function to define the static low-order FC. Static FC network is an extreme case
    where window length is maximized to the entire timescale.

    :param signal: The extracted ROI signals
    :return: The computed static low-order FC
    """
    return dynamic_low_order_fc(signal, int(get_env("SIGNAL_LENGTH")), 1)


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    p = Path(
        "E:\\Adel\\University\\PhD\\MECO\\INSPIRE\\Data\\ADNI\\Signals\\"
        "full-band_ROISignals_FunImgARCWSF\\ROISignals_002_S_1155-2012_12_20.npz"
    )
    signal_ = np.load(p)["signal"]
    coeff = dynamic_low_order_fc(signal_, window_length=70, stride=1)
    ic(coeff.shape)
    ic(coeff.min())
    ic(coeff.max())

    pd.DataFrame(coeff[10]).to_csv(
        get_env("PROJECT_ROOT") + "\\dynamic_low_order_fc_10.csv"
    )

    fig, ax = plt.subplots()
    sns.heatmap(coeff[10], cmap="RdYlBu_r", ax=ax)
    ax.axis("off")
    plt.show()
