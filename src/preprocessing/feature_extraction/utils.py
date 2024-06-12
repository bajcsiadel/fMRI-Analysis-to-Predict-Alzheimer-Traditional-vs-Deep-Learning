import numpy as np
from icecream import ic


class SlidingWindow:
    """
    Represents a sliding window with the specified length and stride.
    It will be applied vertically.

    :param window_length: size of the window
    :param stride: size of the step. Defaults to ``1``.
    """
    def __init__(self, window_length: int | tuple, stride: int = 1):
        self.__window_length = window_length
        self.__stride = stride

    def apply(self, array: np.ndarray) -> np.ndarray:
        """
        Applies the sliding window for the specified array.

        :param array:
        :return: the generated sub-series using the specified window
        :raises ValueError: if the given array is not 2-dimensional
        """
        if len(array.shape) != 2:
            raise ValueError("Array must be 2-dimensional")
        result_shape = [
            *(
                np.array(array.shape) - np.array([self.__window_length, 0])
             ) // np.array([self.__stride]),
            self.__window_length,
        ]
        result_shape[0] += 1
        result = np.empty(result_shape)

        for j in np.arange(array.shape[1]):
            for i in np.arange(
                    0, array.shape[0] - self.__window_length + 1, self.__stride
            ):
                result[i, j, :] = array[i: i + self.__window_length, j]
        return result

    def __call__(self, *args, **kwargs):
        return self.apply(*args)


def covariance(x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
    x = x.copy()
    if y is None:
        y = x.copy()

    if x.shape != y.shape:
        raise ValueError

    x -= x.mean(axis=1)[:, np.newaxis]
    y -= y.mean(axis=1)[:, np.newaxis]

    c = x @ y.T
    c *= (1 / (x.shape[1] - 1))

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
    window = SlidingWindow(window_length=3, stride=1)

    array = np.arange(100).reshape(20, 5)

    sub_series = window.apply(array)
    ic(sub_series.shape)
    ic(sub_series[0])
    test_covariance()
    test_correlation()
