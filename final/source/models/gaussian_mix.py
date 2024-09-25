import numpy as np

class BayesianGaussianMixture:
    classes: np.ndarray
    class_priors: np.ndarray
    class_means: np.ndarray
    class_covs: np.ndarray
    class_weights: np.ndarray  # pesos das misturas

    def __init__(self, num_components=2) -> None:
        self.name = "GaussianMixture"
        self.num_components = num_components  # número de gaussianas por classe

    def fit(self, train):
        self.classes = np.unique(train[:, -1])
        n_features = train.shape[1] - 1

        self.class_priors = np.zeros(len(self.classes))
        self.class_means = np.zeros((len(self.classes), self.num_components, n_features))
        self.class_covs = np.zeros((len(self.classes), self.num_components, n_features, n_features))
        self.class_weights = np.ones((len(self.classes), self.num_components)) / self.num_components  # inicializa pesos uniformemente

        for i, c in enumerate(self.classes):
            data = train[train[:, -1] == c][:, :-1]
            self.class_priors[i] = len(data) / len(train)

            # Inicialização dos parâmetros da mistura de gaussianas (GMM) para cada classe
            for k in range(self.num_components):
                subset = data[np.random.choice(data.shape[0], size=data.shape[0] // self.num_components, replace=False)]
                self.class_means[i, k] = np.mean(subset, axis=0)
                self.class_covs[i, k] = np.cov(subset.T)

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
                likelihood = self._mixture_gaussian_likelihood(data, i)

                # cálculo da probabilidade posterior
                posteriors[i] = prior * likelihood

            prediction = self.classes[np.argmax(posteriors)]
            predictions.append((newData, prediction))

        return predictions

    def _multivariate_gaussian(self, x, mean, cov):
        cov += np.eye(cov.shape[0]) * 1e-6  # regularização
        exp_term = np.exp(-0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean)))
        likelihood = exp_term / np.sqrt(np.linalg.det(cov) * (2 * np.pi) ** (len(x) / 2))
        return likelihood

    def _mixture_gaussian_likelihood(self, x, class_idx):
        likelihood = 0
        for k in range(self.num_components):
            mean = self.class_means[class_idx, k]
            cov = self.class_covs[class_idx, k]
            weight = self.class_weights[class_idx, k]

            likelihood += weight * self._multivariate_gaussian(x, mean, cov)
        return likelihood

    def __copy__(self):
        classifier = BayesianGaussianMixture(num_components=self.num_components)
        classifier.classes = self.classes.copy()
        classifier.class_priors = self.class_priors.copy()
        classifier.class_means = self.class_means.copy()
        classifier.class_covs = self.class_covs.copy()
        classifier.class_weights = self.class_weights.copy()
        return classifier
