import numpy as np


def euclidean_distance(self, data1, data2):
    """
    Calculate the Euclidean distance between two sets of data.

    Args:
        data1 (np.ndarray): The first set of data.
        data2 (np.ndarray): The second set of data.

    Returns:
        np.ndarray: An array of the Euclidean distances between the two sets of data.
    """
    return np.sqrt(np.sum((data1 - data2) ** 2, axis=1))
