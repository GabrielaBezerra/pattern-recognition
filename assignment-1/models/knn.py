import numpy as np

class KNNClassifier:

    memory: list[np.ndarray]

    def __init__(self, k: int):
        self.k = k

    def fit(self, feats: np.ndarray, labels: np.ndarray):
        print("fit")
        self.memory = [feats, labels]
        print("memory", self.memory)

    def predict(self, feats: np.ndarray, labels: np.ndarray):
        print("predict", self.k)
        predictions = []
        test: list[np.ndarray] = [feats, labels]
        for newData in test[0]:
            distances = []
            for data in self.memory:
                distances.append(self.euclidean_distance(data, newData))
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = labels[nearest_indices]
            predicted_label = np.bincount(nearest_labels).argmax()
            predictions.append(predicted_label)
        print(predictions)
        return predictions


    def euclidean_distance(self, data1, data2):
        return np.sqrt(np.sum((data1 - data2)**2))
