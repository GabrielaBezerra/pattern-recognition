from methods import experiments
from methods.split import Holdout
from methods.metrics import ClassifierMetrics
from models.knn import KNNClassifier
from models.dmc import DMCClassifier
from utils.logs import Logs
import numpy as np

log = Logs(verbose=True)

for exp in experiments.main:
    log.database(exp.database_name)

    knn = KNNClassifier(k=int(np.sqrt(len(exp.df))))
    log.model(knn.name, exp.database_name)
    exp.realizations(
        model=knn,
        split=Holdout(train_percent=0.8),
        times=20,
        metrics=ClassifierMetrics(),
        log=log,
    )

    dmc = DMCClassifier()
    log.model(dmc.name, exp.database_name)
    exp.realizations(
        model=dmc,
        split=Holdout(train_percent=0.8),
        times=20,
        metrics=ClassifierMetrics(),
        log=log,
    )
