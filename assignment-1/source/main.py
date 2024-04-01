from methods import experiments, split
from utils import databases
from models import knn, dmc

df = databases.loadIris()#.groupby("Species").head(10)

experiments.realizations(
    df, split_method=split.Holdout(train_percent=0.8), model=knn.KNNClassifier(k=10), times=20
)

# experiments.realizations(
#     df, split_method="holdout", split_proportion=0.7, model=dmc.DMCClassifier()
# )