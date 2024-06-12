import numpy as np
from icecream import ic

from preprocessing.feature_extraction.utils import SlidingWindow, correlation
from utils.environment import get_env


def dynamic_low_order_fc(
        signal: np.ndarray, window_length: int, stride: int = 1
) -> np.ndarray:
    """
    Function to represent a low order feature extractor presented in [Zhang-2017]_.
    After the sub-series are defined a correlation matrix is constructed using
    Pearson's correlation. In the original article the following values were used:

    >>> window_length = 70
    >>> stride = 1

    .. [Zhang-2017] Zhang, Y., Zhang, H., Chen, X., Lee, S.-W., & Shen, D. (2017).
        Hybrid High-order Functional Connectivity Networks Using Resting-state
        Functional MRI for Mild Cognitive Impairment Diagnosis. Scientific Reports,
        7(1), 6530. https://doi.org/10.1038/s41598-017-06509-0

    :param signal: The extracted ROI signals
    :param window_length: The length of the sliding window.
    :param stride: The stride used for the sliding window. Defaults to ``1``.
    :return: The computed dynamic low-order FC
    """
    sliding_window = SlidingWindow(window_length, stride)
    X = sliding_window(signal)
    # defining correlation
    C = np.empty((*X.shape[:2], X.shape[1]))
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
    return dynamic_low_order_fc(signal, get_env("TR"), 1)


if __name__ == '__main__':
    from pathlib import Path

    import pandas as pd
    # import seaborn as sns

    p = Path("C:\\Users\\User\\Documents\\My-staff\\Datasets\\ADNI-structured\\"
             "top-down_3\\Results\\slow4_ROISignals_FunImgARCWSF\\"
             "ROISignals_002_S_0295-2011_06_02.npz")
    signal_ = np.load(p)["signal"]
    coeff = dynamic_low_order_fc(signal_, window_length=70, stride=1)
    ic(coeff.shape)
    ic(coeff.min())
    ic(coeff.max())

    pd.DataFrame(coeff[10]).to_csv(
        get_env("PROJECT_ROOT") + "\\dynamic_low_order_fc_10.csv"
    )

    # sns.heatmap(coeff)
