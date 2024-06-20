import numpy as np
from pipe import *


@Pipe
def to_list(x):
    return list(x)


@Pipe
def to_numpy(x):
    if type(x) is not list:
        x = list(x)
    return np.array(x)
