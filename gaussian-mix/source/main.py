from methods import experiments
from methods.split import Holdout
from models.knn import KNNClassifier
from models.dmc import DMCClassifier
from models.bayesian_gaussian_multivariate import BayesianGaussianMultivariate
from models.naive_bayes import NaiveBayesClassifier
from models.bayesian_gaussian_linear_quadratic import BayesianGaussianLinear, BayesianGaussianQuadratic
from models.kmedias import KMeans
from models.gaussian_mix import BayesianGaussianMixture
from utils.plot import Plot
from utils import log
import numpy as np

log.verbose = False

for exp in experiments.main:
    log.database(exp.database_name)
    exp.realizations(
        models=[
            KNNClassifier(k=int(np.sqrt(len(exp.df)))),
            DMCClassifier(),
            BayesianGaussianMultivariate(),
            NaiveBayesClassifier(),
            BayesianGaussianLinear(),
            BayesianGaussianQuadratic(),
            KMeans(),
            BayesianGaussianMixture(),
        ],
        split=Holdout(train_percent=0.7),
        times=1,
        plots=[
            Plot.TRAIN_TEST,
            Plot.DECISION_BOUNDARY,
            Plot.DECISION_BOUNDARY_3D,
            Plot.GAUSSIAN_CURVES_3D,
        ],
        plot_delay=0.1,
    )
