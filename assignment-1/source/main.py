from socket import sethostname
import numpy as np
import pandas as pd
from methods import experiments
from models import knn, dmc

df = pd.read_csv("datasets/iris/Iris.csv")

#       Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm         Species
# 0      1            5.1           3.5            1.4           0.2     Iris-setosa
# 1      2            4.9           3.0            1.4           0.2     Iris-virginca
# 2      3            4.7           3.2            1.3           0.2     Iris-versicolor

# [150 rows x 6 columns]

# filter df to get 10 rows where Species is Iris-setosa, Iris-virginica, and Iris-versicolor
df = pd.concat(
    [
        df[df["Species"] == "Iris-setosa"].head(10),
        df[df["Species"] == "Iris-virginica"].head(10),
        df[df["Species"] == "Iris-versicolor"].head(10),
    ]
)

experiments.realizations(
    df, split_method="holdout", split_proportion=0.7, model=knn.KNNClassifier(k=5)
)
# realizations(df, dmc)
