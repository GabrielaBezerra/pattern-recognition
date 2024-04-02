from methods import experiment
from methods.split import Holdout
from utils import databases
from models.knn import KNNClassifier
from models.dmc import DMCClassifier

for m in [KNNClassifier(k=5), DMCClassifier()]:
    experiment.realizations(
        df=databases.loadColumn(),
        split_method=Holdout(train_percent=0.8), 
        model=m,
        times=20, 
        verbose=True
    )