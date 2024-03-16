# deboraruth.me instagram pefoce

import pandas as pd
from methods import split, metric
from models import knn, dmc

df = pd.read_csv("datasets/iris/Iris.csv")

#       Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm         Species
# 0      1            5.1           3.5            1.4           0.2     Iris-setosa
# 1      2            4.9           3.0            1.4           0.2     Iris-setosa
# 2      3            4.7           3.2            1.3           0.2     Iris-setosa
# 3      4            4.6           3.1            1.5           0.2     Iris-setosa
# 4      5            5.0           3.6            1.4           0.2     Iris-setosa
# ..   ...            ...           ...            ...           ...             ...
# 145  146            6.7           3.0            5.2           2.3  Iris-virginica
# 146  147            6.3           2.5            5.0           1.9  Iris-virginica
# 147  148            6.5           3.0            5.2           2.0  Iris-virginica
# 148  149            6.2           3.4            5.4           2.3  Iris-virginica
# 149  150            5.9           3.0            5.1           1.8  Iris-virginica

# [150 rows x 6 columns]


def realizations(df: pd.DataFrame, model):
    # metrics
    metric_set = metric.ClassifierMetric(label="setosa")
    metric_ver = metric.ClassifierMetric(label="versicolor")
    metric_vir = metric.ClassifierMetric(label="virginica")

    for i in range(0, 10):
        train, train_labels, test, test_labels = split.holdout(df[:10])
        model.fit(train.to_numpy(), train_labels.to_numpy())
        predictions = model.predict(test.to_numpy(), test_labels.to_numpy())
        # map predictions to label_hit_miss
        versicolor_hit_miss = [1, 0, 0]
        virginica_hit_miss = [1, 0, 1]
        setosa_hit_miss = [1, 1, 1]

        metric_ver.compute(versicolor_hit_miss)
        metric_vir.compute(virginica_hit_miss)
        metric_set.compute(setosa_hit_miss)


realizations(df, knn.KNNClassifier(k=5))
# realizations(df, dmc)
