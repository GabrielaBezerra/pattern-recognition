from typing import Counter
import numpy as np


class KNNClassifier:
    memory: np.ndarray

    def __init__(self, k: int):
        self.k = k

    def fit(self, train: np.ndarray):
        print("fit in train\t", len(train))
        self.memory = train

    # memory
    # [[1, 2, 3, 1],
    #   [7, 8, 9, 0],
    #   [4, 5, 6, 1],
    #   [14, 5, 1, 0]]
    # write memory in one line, keeping its shape
    # [[1, 2, 3, 1], [7, 8, 9, 0], [4, 5, 6, 1], [14, 5, 1, 0]]
    # distances from [1, 2, 3] to each row in memory excluding the last column
    # [ 0,  9.89949494,  3.60555128, 13.92838828]
    # sorted_indices from distances
    # [0, 2, 1, 3]
    # 3 nearest_indices from sorted_indices
    # [0, 2, 1]
    # nearest_labels from nearest_indices, considering label the last column of memory
    # [1, 1, 0]
    # bincount of nearest_labels
    # [1, 2]
    # most frequent / predicted label
    # 1
    def predict(self, test: np.ndarray):
        print("predict in test\t", len(test))
        print("k =", self.k)
        predictions = []  # [(newData, prediction), ...]
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
        # convert data1 and data2 to float type
        data1 = data1.astype(float)
        data2 = data2.astype(float)
        return np.sqrt(np.sum((data2 - data1) ** 2, axis=1))

    def hitOrMiss(self, prediction: tuple):
        expectedLabel = prediction[0][-1]
        predictedLabel = prediction[1]
        return 1 if expectedLabel == predictedLabel else 0
