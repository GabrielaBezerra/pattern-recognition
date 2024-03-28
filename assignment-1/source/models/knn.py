import numpy as np
from collections import Counter


class KNNClassifier:
    memory: np.ndarray

    def __init__(self, k):
        """
        Initialize the KNNClassifier.

        Args:
            k (int): The number of nearest neighbors to consider.
        """
        self.k = k

    def fit(self, train):
        """
        Fit the KNNClassifier with the training data.

        Args:
            train (np.ndarray): The training data.
        """
        print("fit in train\t", len(train))
        self.memory = train

    def predict(self, test):
        """
        Predict the labels for the test data.

        Args:
            test (np.ndarray): The test data.

        Returns:
            list[tuple[np.ndarray, str]]: A list of tuples containing the test data and the predicted labels.
        """
        print("predict in test\t", len(test))
        print("k =", self.k)
        predictions: list[tuple[np.ndarray, str]] = []  # [(newData, prediction), ...]
        for newData in test:
            distances = self.euclidean_distance(self.memory[:, :-1], newData[:-1])
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

    def euclidean_distance(self, data1, data2):
        """
        Calculate the Euclidean distance between two sets of data.

        Args:
            data1 (np.ndarray): The first set of data.
            data2 (np.ndarray): The second set of data.

        Returns:
            np.ndarray: The Euclidean distance between the two sets of data.
        """
        # convert data1 and data2 to float type
        data1 = data1.astype(float)
        data2 = data2.astype(float)
        return np.sqrt(np.sum((data2 - data1) ** 2, axis=1))

    def hitOrMiss(self, prediction):
        """
        Check if the predicted label matches the expected label.

        Args:
            prediction (tuple): A tuple containing the expected label and the predicted label.

        Returns:
            int: 1 if the labels match, 0 otherwise.
        """
        expectedLabel = prediction[0][-1]
        predictedLabel = prediction[1]
        return 1 if expectedLabel == predictedLabel else 0
