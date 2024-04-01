import numpy as np
from collections import Counter
from methods.distances import euclidean_distance


class DMCClassifier:
    memory: np.ndarray

    def fit(self, train):
        """
        Fit the DMCClassifier with the training data.

        Args:
            train (np.ndarray): The training data.
        """
        print("fit in train\t", len(train))
        # compute centroids
        centroids = {}
        for row in train:
            label = row[-1]
            if label not in centroids:
                centroids[label] = []
            centroids[label].append(row[:-1])
        for label in centroids:
            centroids[label] = np.mean(centroids[label])
        self.memory = centroids

    def predict(self, test):
        """
        Predict the labels for the test data.

        Args:
            test (np.ndarray): The test data.

        Returns:
            list[tuple[np.ndarray, str]]: A list of tuples containing the test data and the predicted labels.
        """
        print("predict in test\t", len(test))
        predictions: list[tuple[np.ndarray, str]] = []  # [(newData, prediction), ...]
        for newData in test:
            for label in self.memory.keys():
                distances = euclidean_distance(self.memory[label], newData[:-1])
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