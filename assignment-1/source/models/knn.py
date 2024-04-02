import numpy as np
from collections import Counter
from utils.distances import euclidean_distance


class KNNClassifier:
    memory: np.ndarray

    def __init__(self, k):
        """
        Initialize the KNNClassifier.

        Args:
            k (int): The number of nearest neighbors to consider.
        """
        self.name = "KNN"
        self.k = k

    def fit(self, train):
        """
        Fit the KNNClassifier with the training data.

        Args:
            train (np.ndarray): The training data.
        """
        self.memory = train

    def predict(self, test, has_labels=True):
        """
        Predict the labels for the test data.

        Args:
            test (np.ndarray): The test data.

        Returns:
            list[tuple[np.ndarray, str]]: A list of tuples containing the test data and the predicted labels.
        """
        predictions: list[tuple[np.ndarray, str]] = []  # [(newData, prediction), ...]
        for newData in test:
            if has_labels:
                distances = euclidean_distance(self.memory[:, :-1], newData[:-1])
            else:
                distances = euclidean_distance(self.memory[:, :-1], newData)
            sorted_indices = np.argsort(distances)
            nearest_indices = sorted_indices[: self.k]
            # get memory rows from array of nearest indices
            nearest_labels = self.memory[nearest_indices]
            # convert nearest_labels to int type
            nearest_labels = nearest_labels
            # get the most frequent label
            nearest_labels_count = Counter(nearest_labels[:, -1])
            prediction = nearest_labels_count.most_common(1)[0][0]
            predictions.append((newData, prediction))
        return predictions
