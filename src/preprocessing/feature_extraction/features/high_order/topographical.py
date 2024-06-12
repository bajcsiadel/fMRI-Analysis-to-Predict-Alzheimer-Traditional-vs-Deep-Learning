import numpy as np
from icecream import ic

from preprocessing.feature_extraction.features.low_order import dynamic_low_order_fc
from preprocessing.feature_extraction.utils import correlation
from utils.environment import get_env


def dynamic_topographical_high_order_fc(
        *,
        C: np.ndarray = None,
        signal: np.array = None, window_length: int = None, stride: int = None
) -> np.ndarray:
    """
    Dynamic high-order FC network can be constructed by calculating the FC between
    every pair of the low-order sub-networks (one line from the low-order FC
    representing the connectivity of a selected ROI to all others).

    >>> window_length = 70
    >>> stride = 1

    .. [Zhang-2017] Zhang, Y., Zhang, H., Chen, X., Lee, S.-W., & Shen, D. (2017).
        Hybrid High-order Functional Connectivity Networks Using Resting-state
        Functional MRI for Mild Cognitive Impairment Diagnosis. Scientific Reports,
        7(1), 6530. https://doi.org/10.1038/s41598-017-06509-0

    :param C: The computed low-order FC. Defaults to ``None``.
    :param signal: The extracted ROI signals. Defaults to ``None``.
    :param window_length: The length of the sliding window. Defaults to ``None``.
    :param stride: The stride used for the sliding window. Defaults to ``None``.
    :return: dynamic high-order FC
    :raises AttributeError: if C is not given and have to be computed by either
    `signal`, `window_length` or `stride` is missing.
    :raises ValueError: if C does not have the correct shape.
    """
    if C is None:
        other_nones = [variable is None for variable in [signal, window_length, stride]]
        if any(other_nones):
            raise AttributeError(
                "If C is not given `signal`, `window_length` "
                "and `stride` must be given."
            )
        C = dynamic_low_order_fc(signal, window_length, stride)
    elif len(C.shape) != 3:
        raise ValueError(f"C must be 3-dimensional, but {len(C.shape)} got.")

    H = np.empty_like(C)
    for k in range(C.shape[0]):
        H[k] = correlation(C[k])

    return H


def static_topographical_high_order_fc(
        *, C: np.ndarray = None, signal: np.array = None
) -> np.ndarray:
    """
    Static high-order FC network can be constructed by calculating the FC between
    every pair of the low-order sub-networks (one line from the low-order FC
    representing the connectivity of a selected ROI to all others).

    .. [Zhang-2017] Zhang, Y., Zhang, H., Chen, X., Lee, S.-W., & Shen, D. (2017).
        Hybrid High-order Functional Connectivity Networks Using Resting-state
        Functional MRI for Mild Cognitive Impairment Diagnosis. Scientific Reports,
        7(1), 6530. https://doi.org/10.1038/s41598-017-06509-0

    :param C: The computed low-order FC. Defaults to ``None``.
    :param signal: The extracted ROI signals. Defaults to ``None``.
    :return: dynamic high-order FC
    :raises AttributeError: if C is not given and have to be computed by either
    `signal`, `window_length` or `stride` is missing.
    :raises ValueError: if C does not have the correct shape.
    """
    return dynamic_topographical_high_order_fc(
        C=C, signal=signal, window_length=get_env("TR"), stride=1
    )


if __name__ == '__main__':
    from pathlib import Path

    import pandas as pd
    # import seaborn as sns

    p = Path("C:\\Users\\User\\Documents\\My-staff\\Datasets\\ADNI-structured\\"
             "top-down_3\\Results\\slow4_ROISignals_FunImgARCWSF\\"
             "ROISignals_002_S_0295-2011_06_02.npz")
    signal_ = np.load(p)["signal"]
    coeff = dynamic_topographical_high_order_fc(signal=signal_, window_length=70, stride=1)
    ic(coeff.shape)
    ic(coeff.min())
    ic(coeff.max())

    pd.DataFrame(coeff[10]).to_csv(
        get_env("PROJECT_ROOT") + "\\dynamic_traditional_high_order_fc_10.csv"
    )

    # sns.heatmap(coeff)
