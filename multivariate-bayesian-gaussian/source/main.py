from methods import experiments
from methods.split import Holdout
from methods.metrics import ClassifierMetrics
from models.knn import KNNClassifier
from models.dmc import DMCClassifier
from models.bayesian_gaussian_multivariate import BayesianGaussianMultivariate
from utils import log
import numpy as np

log.verbose = False

for exp in experiments.main:
    log.database(exp.database_name)

    knn = KNNClassifier(k=int(np.sqrt(len(exp.df))))
    log.model(knn.name, exp.database_name)
    exp.realizations(
        model=knn,
        split=Holdout(train_percent=0.7),
        times=20,
        metrics=ClassifierMetrics(),
        plot_train_test=False,
        plot_decision_boundary=False,
        plot_delay=0.5,
    )

    dmc = DMCClassifier()
    log.model(dmc.name, exp.database_name)
    exp.realizations(
        model=dmc,
        split=Holdout(train_percent=0.7),
        times=20,
        metrics=ClassifierMetrics(),
        plot_train_test=False,
        plot_decision_boundary=False,
        plot_delay=0.5,
    )

    bayesGausMulti = BayesianGaussianMultivariate()
    log.model(bayesGausMulti.name, exp.database_name)
    exp.realizations(
        model=bayesGausMulti,
        split=Holdout(train_percent=0.7),
        times=20,
        metrics=ClassifierMetrics(),
        plot_train_test=False,
        plot_decision_boundary=False,
        plot_gaussians=False,
        plot_delay=0.5,
    )
