import numpy as np
from icecream import ic

from preprocessing.feature_extraction.features.high_order.topographical import (
    dynamic_topographical_high_order_fc,
    static_topographical_high_order_fc,
)
from preprocessing.feature_extraction.features.low_order import (
    dynamic_low_order_fc,
    static_low_order_fc,
)
from preprocessing.feature_extraction.helpers import correlation, normalize
from utils.environment import get_env


def dynamic_associated_high_order_fc(
    *,
    C: np.ndarray = None,
    H: np.ndarray = None,
    signal: np.array = None,
    window_length: int = None,
    stride: int = None,
) -> np.ndarray:
    """
    Dynamic high-order FC network can be constructed by calculating the FC between
    every pair of the low-order sub-networks (one line from the low-order FC
    representing the connectivity of a selected ROI to all others). In the original
    article the following values were used: window_length = 70 and stride = 1.


    .. [Zhang-2017] Zhang, Y., Zhang, H., Chen, X., Lee, S.-W., & Shen, D. (2017).
        Hybrid High-order Functional Connectivity Networks Using Resting-state
        Functional MRI for Mild Cognitive Impairment Diagnosis. Scientific Reports,
        7(1), 6530. https://doi.org/10.1038/s41598-017-06509-0

    :param C: The computed low-order FC. Defaults to ``None``.
    :param H: The computed high-order FC. Defaults to ``None``.
    :param signal: The extracted ROI signals. Defaults to ``None``.
    :param window_length: The length of the sliding window. Defaults to ``None``.
    :param stride: The stride used for the sliding window. Defaults to ``None``.
    :returns: dynamic associated high-order FC
    :raises AttributeError: if C and H are not given and either `signal`,
    `window_length` or `stride` is missing.
    :raises ValueError: if C or H are None
    :raises ValueError: if C and H are not 3-dimensional
    :raises ValueError: if C or H do not have the same shape
    """
    if C is None and H is None:
        other_nones = [variable is None for variable in [signal, window_length, stride]]
        if any(other_nones):
            raise AttributeError(
                "If C and H are not given `signal`, `window_length` "
                "and `stride` must be given."
            )
        C = dynamic_low_order_fc(signal, window_length, stride)
        H = dynamic_topographical_high_order_fc(C=C)

    if C is None or H is None:
        raise ValueError("Both C and H must be specified.")
    elif len(C.shape) != 3:
        raise ValueError(f"C must be 3-dimensional, but {len(C.shape)} got.")
    elif len(H.shape) != 3:
        raise ValueError(f"H must be 3-dimensional, but {len(H.shape)} got.")
    elif C.shape != H.shape:
        raise ValueError(
            f"C and H must have the same shape, but {C.shape} and {H.shape} got."
        )

    C = normalize(C)
    H = normalize(H)

    A = np.empty_like(C)
    for k in range(C.shape[0]):
        A[k] = C[k].T @ H[k]
        A[k] = (A[k] + A[k].T) / 2
    A /= C.shape[1]
    return A


def static_associated_high_order_fc(
    *, C: np.ndarray = None, H: np.ndarray = None, signal: np.array = None
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
    :param H: The computed high-order FC. Defaults to ``None``.
    :param signal: The extracted ROI signals. Defaults to ``None``.
    :returns: static associated high-order FC
    :raises AttributeError: if C and H are not given and `signal` is missing.
    """
    if C is None and H is None:
        if signal is None:
            raise AttributeError("If C and H are not given `signal` must be given.")
        C = static_low_order_fc(signal)
        H = static_topographical_high_order_fc(C=C)

    return dynamic_associated_high_order_fc(C=C, H=H)


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
    coeff = dynamic_associated_high_order_fc(signal=signal_, window_length=70, stride=1)
    ic(coeff.shape)
    ic(coeff.min())
    ic(coeff.max())

    pd.DataFrame(coeff[10]).to_csv(
        get_env("PROJECT_ROOT") + "\\dynamic_associated_high_order_fc_10.csv"
    )

    fig, ax = plt.subplots()
    sns.heatmap(coeff[10], cmap="RdYlBu_r", ax=ax)
    ax.axis("off")
    plt.show()
