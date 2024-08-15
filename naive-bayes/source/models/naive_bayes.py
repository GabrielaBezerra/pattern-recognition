import numpy as np

# Explicação: https://medium.com/@rangavamsi5/na%C3%AFve-bayes-algorithm-implementation-from-scratch-in-python-7b2cc39268b9

# 1 Modelo
# Calcular verossimilhança
# Calcular probabilidades a posteiriori para cada uma das classes
# Treinar no Iris, Coluna, Breast Cancer, Dermatology e Artificial II

# 2 Experiment
# 20 realizacoes no minimo

# 3 Metrics
# Computar acuracia e desvio
# Comparar com KNN e DMC
# Apresentar Matriz de Confusão para uma realizacao, justificando escolha.

# 4 Plot
# Mostrar superficie de decisão (Artificial II, Coluna). Escolha atributos e justifique.
# Mostrar Gaussiana para cada classe (Artificial II, Coluna) e conjuntos de dados de treino e teste para a realização que foi escolhida pra plotar a Gaussiana.

# Relatório comparando com modelos anteriores.


# Naive Bayes
class NaiveBayesClassifier:
    classes: np.ndarray
    class_priors: np.ndarray
    class_means: np.ndarray
    class_vars: np.ndarray

    def __init__(self) -> None:
        self.name = "NaiveBayes"

    def fit(self, train):
        self.classes = np.unique(train[:, -1])

        self.class_priors = np.zeros(len(self.classes))
        self.class_means = np.zeros((len(self.classes), train.shape[1] - 1))
        self.class_vars = np.zeros((len(self.classes), train.shape[1] - 1))

        for i, c in enumerate(self.classes):
            data = train[train[:, -1] == c][:, :-1]

            self.class_priors[i] = len(data) / len(train)
            self.class_means[i] = np.mean(data, axis=0)
            self.class_vars[i] = np.var(data, axis=0)

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
                var = self.class_vars[i]

                # calculating the likelihood of the data point under the class's Gaussian distribution
                likelihood = self._univariate_gaussian(data, mean, var)

                # calculating the posterior probability
                posteriors[i] = prior * likelihood

            # choosing the class with the highest posterior probability as the prediction
            prediction = self.classes[np.argmax(posteriors)]
            predictions.append((newData, prediction))

        return predictions

    def _univariate_gaussian(self, x, mean, var):
        # apply regularization to the covariance matrix to avoid singular matrix
        var += 1e-6

        # calculating gaussian distributions for each feature
        gaussians = (1 / np.sqrt(2 * np.pi * var)) * np.exp(
            -((x - mean) ** 2) / (2 * var)
        )

        # returning likelihood
        return np.prod(gaussians)

    def __copy__(self):
        classifier = NaiveBayesClassifier()
        classifier.classes = self.classes.copy()
        classifier.class_priors = self.class_priors.copy()
        classifier.class_means = self.class_means.copy()
        classifier.class_vars = self.class_vars.copy()
        return classifier
