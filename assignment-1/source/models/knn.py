import numpy as np
from collections import Counter
from utils.distances import euclidean_distance


class KNNClassifier:
    memory: np.ndarray

    def __init__(self, k):
        self.name = "KNN"
        self.k = k

    def fit(self, train):
        self.memory = train

    def predict(self, test, has_labels=True):
        predictions: list[tuple[np.ndarray, str]] = []  # [(newData, prediction), ...]
        for newData in test:
            if has_labels:
                distances = euclidean_distance(self.memory[:, :-1], newData[:-1])
            else:
                distances = euclidean_distance(self.memory[:, :-1], newData)
            sorted_indices = np.argsort(distances)
            nearest_indices = sorted_indices[: self.k]
            nearest_labels = self.memory[nearest_indices]
            nearest_labels = nearest_labels
            nearest_labels_count = Counter(nearest_labels[:, -1])
            prediction = nearest_labels_count.most_common(1)[0][0]
            predictions.append((newData, prediction))
        return predictions
