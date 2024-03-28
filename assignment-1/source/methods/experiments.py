import pandas as pd
from methods import split, metric
from models import knn, dmc


def realizations(df: pd.DataFrame, model, split_method, split_proportion=0.7, times=10):
    """
    Perform multiple realizations of a pattern recognition experiment.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        model: The pattern recognition model to be used.
        split_method (str): The method for splitting the dataset into training and testing sets.
            Currently supported methods are "holdout", "kfold", and "leaveoneout".
        split_proportion (float, optional): The proportion of the dataset to be used for training
            when using the "holdout" split method. Defaults to 0.7.
        times (int, optional): The number of realizations to perform. Defaults to 10.
    """
    # metrics
    metric_set = metric.ClassifierMetric(label="setosa")
    metric_ver = metric.ClassifierMetric(label="versicolor")
    metric_vir = metric.ClassifierMetric(label="virginica")

    for i in range(1, times + 1):
        all_predictions = []
        print("\n🍵 Realization", i)

        if split_method == "holdout":
            train, test = split.holdout(df, train_percent=split_proportion)
        # Requires python 3.10 and later
        # match split_method:
        #     case "holdout":
        #         train, test = split.holdout(df)
        #     # case "kfold":
        #     #     train, test = split.kfold(df, k=5)
        #     # case "leaveoneout":
        #     #     train, test = split.leaveoneout(df)
        #     case _:
        #         raise ValueError("Invalid split method")

        model.fit(train.to_numpy())

        predictions = model.predict(test.to_numpy())
        for pred in predictions:
            if pred[0][-1] != pred[1]:
                print(f"MISS {pred[1]}")
        all_predictions.append(predictions)

        # show confusion matrix for realization
        confusion_matrix = pd.crosstab(
            [prediction[1] for prediction in predictions],
            [prediction[0][-1] for prediction in predictions],
        )
        print(f"Confusion Matrix - Realization {i} \n{confusion_matrix}")

        # # compute metrics for realization
        # # TODO FIX: make hit rate from confusion matrix
        # metric_ver.compute(versicolor_hit_miss)
        # metric_vir.compute(virginica_hit_miss)
        # metric_set.compute(setosa_hit_miss)

    # print(
    #     f"\nAccuracy: {(np.average(np.array([metric_set.accuracy, metric_ver.accuracy, metric_vir.accuracy]))):.2f}"
    # )
    # print(
    #     f"Standard Deviation: {(np.average(np.array([metric_set.std, metric_ver.std, metric_vir.std]))):.2f}"
    # )

    # print(f"\n# Final Metrics after {times} realizations")
    # TODO: show final confusion matrix with average hit and miss amount of all predictions
