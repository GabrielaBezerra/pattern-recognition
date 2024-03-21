import numpy as np
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


def realizations(df: pd.DataFrame, model, times: int = 10):
    # metrics
    metric_set = metric.ClassifierMetric(label="setosa")
    metric_ver = metric.ClassifierMetric(label="versicolor")
    metric_vir = metric.ClassifierMetric(label="virginica")
    all_predictions = []

    for i in range(1, times + 1):
        print("\n# Realization", i)
        train, test = split.holdout(df)
        model.fit(train.to_numpy())
        predictions = model.predict(test.to_numpy())
        all_predictions.append(predictions)

        # map predictions to label_hit_miss
        versicolor_hit_miss = [
            model.hitOrMiss(prediction)
            for prediction in predictions
            if prediction[0][-1] == "Iris-versicolor"
        ]
        virginica_hit_miss = [
            model.hitOrMiss(prediction)
            for prediction in predictions
            if prediction[0][-1] == "Iris-virginica"
        ]
        setosa_hit_miss = [
            model.hitOrMiss(prediction)
            for prediction in predictions
            if prediction[0][-1] == "Iris-setosa"
        ]

        # compute metrics for realization
        metric_ver.compute(versicolor_hit_miss)
        metric_vir.compute(virginica_hit_miss)
        metric_set.compute(setosa_hit_miss)

        # show confusion matrix for realization
        print(f"\nConfusion Matrix - Realization {i}")
        print(
            pd.crosstab(test["Species"], [prediction[1] for prediction in predictions])
        )

    metric_ver.log()
    metric_vir.log()
    metric_set.log()

    print(
        f"\nAccuracy: {(np.average(np.array([metric_set.accuracy, metric_ver.accuracy, metric_vir.accuracy]))):.2f}"
    )
    print(
        f"Standard Deviation: {(np.average(np.array([metric_set.std, metric_ver.std, metric_vir.std]))):.2f}"
    )

    print(f"\n# Final Metrics after {times} realizations")
    # TODO: show final confusion matrix with average hit and miss amount of all predictions


realizations(df, knn.KNNClassifier(k=5))
# realizations(df, dmc)
