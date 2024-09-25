from methods import experiments
from methods.split import Holdout
from methods.metrics import ClassifierMetrics
from models.knn import KNNClassifier
from models.dmc import DMCClassifier
from models.bayesian_gaussian_multivariate import BayesianGaussianMultivariate
from utils.plot import Plot
from utils import log
import numpy as np

log.verbose = False

for exp in experiments.main:
    log.database(exp.database_name)

    for model in [
        KNNClassifier(k=int(np.sqrt(len(exp.df)))),
        DMCClassifier(),
        BayesianGaussianMultivariate(),
    ]:
        log.model(model.name, exp.database_name)
        exp.realizations(
            model=model,
            split=Holdout(train_percent=0.7),
            times=1,
            metrics=ClassifierMetrics(),
            plots=[
                Plot.TRAIN_TEST,
                Plot.DECISION_BOUNDARY,
                Plot.DECISION_BOUNDARY_3D,
                Plot.GAUSSIAN_CURVES_3D,
            ],
            plot_delay=0,
        )
