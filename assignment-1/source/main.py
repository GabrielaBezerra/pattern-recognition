from methods import databases, experiments
from models import knn, dmc

df = databases.loadIris().groupby("Species").head(10)

# experiments.realizations(
#     df, split_method="holdout", split_proportion=0.7, model=knn.KNNClassifier(k=5)
# )

experiments.realizations(
    df, split_method="holdout", split_proportion=0.7, model=dmc.DMCClassifier()
)