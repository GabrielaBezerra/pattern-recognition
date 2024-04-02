import numpy as np
from utils.distances import euclidean_distance


class DMCClassifier:
    centroids = {}

    def __init__(self):
        """
        Initialize the DMCClassifier.
        """
        self.name = "DMC"

    def fit(self, train):
        """
        Fit the DMCClassifier with the training data.

        Args:
            train (np.ndarray): The training data.
        """
        # compute and store centroids for each class
        for data in train:
            label = data[-1]
            if label not in self.centroids:
                self.centroids[label] = data[:-1] # first data point
            else:
                self.centroids[label] = np.vstack((self.centroids[label], data[:-1])) # acumulate data points
        for label in self.centroids:
            self.centroids[label] = np.average(self.centroids[label], axis=0) # compute the centroid

    def predict(self, test):
        """
        Predict the labels for the test data.

        Args:
            test (np.ndarray): The test data.

        Returns:
            list[tuple[np.ndarray, str]]: A list of tuples containing the test data and the predicted labels.
        """
        predictions: list[tuple[np.ndarray, str]] = []  # [(newData, prediction), ...]
        for newData in test:
            distances = euclidean_distance(np.array(list(self.centroids.values())), newData[:-1])
            nearest_label = list(self.centroids.keys())[np.argmin(distances)]
            predictions.append((newData, nearest_label))
        return predictions