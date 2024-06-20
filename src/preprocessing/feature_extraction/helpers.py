import numpy as np
from icecream import ic


def apply_sliding_window(array: np.ndarray, window_length: int, stride: int = 1) -> np.ndarray:
    """
    Applying a sliding window with the specified length and stride.
    It will be applied horizontally.

    :param array: the given array on which the sliding window will be applied
    :param window_length: size of the window
    :param stride: size of the step. Defaults to ``1``.
    :return: the generated sub-series using the specified window
    :raises ValueError: if the given array is not 2-dimensional
    """
    if len(array.shape) != 2:
        raise ValueError("Array must be 2-dimensional")
    result_shape = [
        *(
            np.array(array.shape) - np.array([window_length, 0])
         ) // np.array([stride]),
        window_length,
    ]
    result_shape[0] += 1
    result = np.empty(result_shape)
    # number of windows, window length, ROIs
    result = np.transpose(result, (0, 2, 1))

    for i in np.arange(0, array.shape[0] - window_length + 1, stride):
        result[i, :, :] = array[i: i + window_length, :]
    return result


def centralize(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x -= x.mean(axis=1)[:, np.newaxis]
    return x


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x = centralize(x)
    x /= x.std(axis=1)[:, np.newaxis]
    return x


def covariance(x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
    x = x.copy()
    if y is None:
        y = x.copy()

    if x.shape != y.shape:
        raise ValueError

    x = normalize(x)
    y = normalize(y)

    c = x.T @ y

    return c.squeeze()


def test_covariance():
    try:
        x = np.arange(15).reshape(5, 3).astype(float)
        assert np.allclose(np.cov(x), covariance(x))

        x = np.arange(15).reshape(3, 5).astype(float)
        assert np.allclose(np.cov(x), covariance(x))

        for _ in range(5):
            rows, cols = np.random.randint(2, 5, size=2)
            x = np.random.randn(rows, cols)
            assert np.allclose(np.cov(x), covariance(x))
    except AssertionError as e:
        ic(e)
        ic(x)
        ic(np.cov(x))
        ic(covariance(x))
        exit()


def correlation(x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
    if y is None:
        y = x.copy()

    cov = covariance(x, y)

    d = np.diag(cov)
    d = np.sqrt(d)

    corr = cov.copy()
    corr /= d[:, np.newaxis]
    corr /= d[np.newaxis, :]

    return corr


def test_correlation():
    try:
        x = np.arange(15).reshape(5, 3).astype(float)
        assert np.allclose(np.corrcoef(x), correlation(x))

        x = np.arange(15).reshape(3, 5).astype(float)
        assert np.allclose(np.corrcoef(x), correlation(x))

        for _ in range(5):
            rows, cols = np.random.randint(2, 5, size=2)
            x = np.random.randn(rows, cols)
            assert np.allclose(np.corrcoef(x), correlation(x))
    except AssertionError as e:
        ic(e)
        ic(x)
        ic(np.corrcoef(x))
        ic(correlation(x))
        exit()


if __name__ == "__main__":
    arr = np.arange(100).reshape(20, 5)

    sub_series = apply_sliding_window(arr, window_length=10, stride=1)
    ic(sub_series.shape)
    ic(sub_series[0])
    test_covariance()
    test_correlation()
