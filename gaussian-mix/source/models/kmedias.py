import numpy as np

class KMeans:
    centroids: np.ndarray
    n_clusters: int
    max_iter: int
    tolerance: float

    def __init__(self, n_clusters=3, max_iter=300, tolerance=1e-4) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.name = "KMeans"

    def fit(self, train):
        # Randomly initialize centroids
        np.random.seed(42)  # For reproducibility
        random_indices = np.random.permutation(train.shape[0])[:self.n_clusters]
        self.centroids = train[random_indices, :-1]  # Exclude labels from centroids
        
        for _ in range(self.max_iter):
            # Assign each point to the nearest centroid
            classifications = self._assign_to_centroids(train)

            # Store the old centroids to check convergence
            old_centroids = self.centroids.copy()

            # Recompute centroids as the mean of points in each cluster
            for i in range(self.n_clusters):
                points_in_cluster = train[classifications == i, :-1]  # Exclude labels
                if len(points_in_cluster) > 0:
                    self.centroids[i] = np.mean(points_in_cluster, axis=0)

            # Check for convergence
            if np.all(np.abs(self.centroids - old_centroids) < self.tolerance):
                break

    def predict(self, test, has_labels=True):
        predictions = []
        for newData in test:
            if has_labels:
                data = newData[:-1]  # Exclude label
            else:
                data = newData
            
            # Find the nearest centroid
            distances = np.linalg.norm(self.centroids - data, axis=1)
            classification = np.argmin(distances)
            
            predictions.append((newData, classification))
        
        return predictions

    def _assign_to_centroids(self, data):
        distances = np.zeros((data.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(data[:, :-1] - centroid, axis=1)  # Exclude labels
        
        # Return the index of the nearest centroid for each data point
        return np.argmin(distances, axis=1)

    def __copy__(self):
        kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, tolerance=self.tolerance)
        kmeans.centroids = self.centroids.copy()
        return kmeans
