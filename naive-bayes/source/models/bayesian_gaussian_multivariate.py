import numpy as np


class BayesianGaussianMultivariate:
    classes: np.ndarray
    class_priors: np.ndarray
    class_means: np.ndarray
    class_covs: np.ndarray

    def __init__(self) -> None:
        self.name = "BayesianGaussianMultivariate"

    def fit(self, train):
        self.classes = np.unique(train[:, -1])

        self.class_priors = np.zeros(len(self.classes))
        self.class_means = np.zeros((len(self.classes), train.shape[1] - 1))
        self.class_covs = np.zeros(
            (len(self.classes), train.shape[1] - 1, train.shape[1] - 1)
        )

        for i, c in enumerate(self.classes):
            data = train[train[:, -1] == c][:, :-1]

            self.class_priors[i] = len(data) / len(train)
            self.class_means[i] = np.mean(data, axis=0)
            self.class_covs[i] = np.cov(data.T)

    def predict(self, test, has_labels=True):
        predictions = []
        for newData in test:
            if has_labels:
                data = newData[:-1]
            else:
                data = newData

            posteriors = np.zeros(len(self.classes))

            for i, _ in enumerate(self.classes):
                prior = self.class_priors[i]
                mean = self.class_means[i]
                cov = self.class_covs[i]

                # calculating the likelihood of the data point under the class's Gaussian distribution
                likelihood = self._multivariate_gaussian(data, mean, cov)

                # calculating the posterior probability
                posteriors[i] = prior * likelihood

            # choosing the class with the highest posterior probability as the prediction
            prediction = self.classes[np.argmax(posteriors)]
            predictions.append((newData, prediction))

        return predictions

    def _multivariate_gaussian(self, x, mean, cov):
        # apply regularization to the covariance matrix to avoid singular matrix
        cov += np.eye(cov.shape[0]) * 1e-6

        # calculating the exponential term of the Gaussian distribution
        exp_term = np.exp(
            -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
        )

        # normalizing by the determinant of the covariance matrix
        likelihood = exp_term / np.sqrt(
            np.linalg.det(cov) * (2 * np.pi) ** (len(x) / 2)
        )
        return likelihood

    def __copy__(self):
        classifier = BayesianGaussianMultivariate()
        classifier.classes = self.classes.copy()
        classifier.class_priors = self.class_priors.copy()
        classifier.class_means = self.class_means.copy()
        classifier.class_covs = self.class_covs.copy()
        return classifier
