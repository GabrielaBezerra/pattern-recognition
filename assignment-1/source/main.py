from methods import experiment
from utils import databases
from utils.plot import Plot
from methods.split import Holdout
from models.knn import KNNClassifier
from models.dmc import DMCClassifier

for model in [KNNClassifier(k=5), DMCClassifier()]:
    experiment.realizations(
        # df=databases.loadColumn(binary=True),
        # df=databases.loadColumn(binary=False),
        df=databases.loadIris(),
        # df=databases.loadArtificial(),
        model=model,
        split=Holdout(train_percent=0.8),
        plot=Plot(feature_a=0, feature_b=1),
        times=1,
        verbose=True,
    )
