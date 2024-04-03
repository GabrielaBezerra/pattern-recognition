from methods import experiments
from methods.split import Holdout
from methods.metrics import ClassifierMetrics
from models.knn import KNNClassifier
from models.dmc import DMCClassifier
from utils.logs import Logs

log = Logs(verbose=False)

models = [KNNClassifier(k=5), DMCClassifier()]

for exp in experiments.all:
    log.database(exp.database_name)

    for model in models:
        log.model(model.name, exp.database_name)

        exp.realizations(
            model=model,
            split=Holdout(train_percent=0.8),
            times=1,
            metrics=ClassifierMetrics(),
            log=log,
        )
