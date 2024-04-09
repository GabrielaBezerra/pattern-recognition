import numpy as np


# The provided Python code defines a class BayesianGaussianMulticlass for a multiclass classification model based on the Bayesian Gaussian method. This method is a type of probabilistic model that uses Bayes' theorem and assumes that the data from each class is drawn from a simple Gaussian distribution. The model calculates the posterior probability of each class for a given data point and makes predictions based on the class with the highest posterior probability. The model is trained on a dataset with multiple classes and can predict the class labels for new data points. The class has methods for fitting the model to the training data and making predictions on new data points.
class BayesianGaussianMulticlass:
    classes: np.ndarray
    class_priors: np.ndarray
    class_means: np.ndarray
    class_covs: np.ndarray

    def __init__(self) -> None:
        self.name = "BayesianGaussianMulticlass"

    # The fit method is used to train the model. It takes a 2D numpy array train as input, where the last column is assumed to be the class labels. The method first identifies the unique classes in the training data and initializes the class priors, means, and covariances to zero. It then calculates the prior probability, mean, and covariance for each class based on the training data. The class priors are the proportion of data points belonging to each class, while the class means and covariances are the mean and covariance of the data points for each class. The method stores these values in the class attributes for later use.
    def fit(self, train):
        # The np.unique function is used to find the unique elements of an array.
        self.classes = np.unique(train[:, -1])
        # The np.zeros function is used to create a new array of given shape and type, filled with zeros.
        self.class_priors = np.zeros(len(self.classes))
        self.class_means = np.zeros((len(self.classes), train.shape[1] - 1))
        self.class_covs = np.zeros(
            (len(self.classes), train.shape[1] - 1, train.shape[1] - 1)
        )
        # The enumerate function is a built-in function of Python.
        for i, c in enumerate(self.classes):
            data = train[train[:, -1] == c][:, :-1]
            self.class_priors[i] = len(data) / len(train)
            self.class_means[i] = np.mean(
                data, axis=0
            )  # The np.mean function is used to calculate the mean of an array along a specified axis.
            self.class_covs[i] = np.cov(
                data.T
            )  # The np.cov function is used to estimate a covariance matrix, given data and weights.

    # The predict method is used to make predictions on new data. It takes a 2D numpy array test as input, where the last column is assumed to be the class labels if has_labels is set to True. For each data point, it calculates the posterior probability for each class by multiplying the class prior and the likelihood of the data point under the class's Gaussian distribution. The class with the highest posterior probability is chosen as the prediction. The method returns a list of tuples containing the new data points and their predicted class labels. The method uses helper methods to calculate the likelihood of a data point under a multivariate Gaussian distribution.
    def predict(self, test, has_labels=True):
        predictions = []
        for newData in test:
            if has_labels:
                data = newData[:-1]
            else:
                data = newData
            posteriors = np.zeros(len(self.classes))
            for i, c in enumerate(self.classes):
                prior = self.class_priors[i]
                mean = self.class_means[i]
                cov = self.class_covs[i]
                likelihood = self._multivariate_gaussian(data, mean, cov)
                posteriors[i] = prior * likelihood
            prediction = self.classes[
                np.argmax(posteriors)
            ]  # The np.argmax function is used to find the indices of the maximum values along an axis.
            predictions.append((newData, prediction))
        return predictions

    # The _multivariate_gaussian method is a helper method used to calculate the likelihood of a data point under a multivariate Gaussian distribution. It takes a data point x, a mean vector mean, and a covariance matrix cov as input, and returns the likelihood of x under the Gaussian distribution defined by mean and cov. The method uses the formula for the multivariate Gaussian distribution to calculate the likelihood. The method is used in the predict method to calculate the likelihood of a data point under each class's Gaussian distribution. The method calculates the exponential term of the Gaussian distribution and normalizes it by the determinant of the covariance matrix.
    def _multivariate_gaussian(self, x, mean, cov):
        x = x - mean
        return np.exp(-0.5 * np.dot(x, np.dot(np.linalg.inv(cov), x))) / np.sqrt(
            np.linalg.det(cov)
        )
