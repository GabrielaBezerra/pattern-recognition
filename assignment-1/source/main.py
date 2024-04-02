from utils import databases
from methods import experiment
from methods.split import Holdout
from models.knn import KNNClassifier
from models.dmc import DMCClassifier

for model in [KNNClassifier(k=5), DMCClassifier()]:
    experiment.realizations(
        df=databases.loadIris(),
        split=Holdout(train_percent=0.8),
        model=model,
        times=20,
        verbose=True
    )