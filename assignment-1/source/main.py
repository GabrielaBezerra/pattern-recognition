from methods import experiment, split
from utils import databases
from models.knn import KNNClassifier
from models.dmc import DMCClassifier

df = databases.loadIris()

for m in [KNNClassifier(k=10), DMCClassifier()]:
    experiment.realizations(
        df, split_method=split.Holdout(train_percent=0.8), model=m, times=20, verbose=True
    )