import numpy as np


def euclidean_distance(data1, data2):
    return np.sqrt(np.sum((data1 - data2) ** 2, axis=1).astype(float))
