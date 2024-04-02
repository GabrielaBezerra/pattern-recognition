import pandas as pd
from .metrics import ClassifierMetrics

def realizations(df: pd.DataFrame, model, split_method, times=10, verbose=True):
    """
    Perform multiple realizations of a pattern recognition experiment.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        model: The pattern recognition model to be used.
        split_method: The method for splitting the dataset into training and testing sets.
        times (int, optional): The number of realizations to perform. Defaults to 10.
        verbose (bool, optional): Prints step by step of the realization execution.
    """

    model_name_for_print = f"\033[1;31m{model.name}\033[0m"
    # print bold white model name
    print(f"\n\033[1;33m=> {model.name} Experiment\033[0m")

    metrics = ClassifierMetrics()

    # Realizations loop
    for i in range(1, times + 1):
        if verbose:
            print(f"\n{model_name_for_print} \033[1;32m#{i} Realization\033[0m")

        train, test = split_method.split(df)
        if verbose:
            print(f"split_method={split_method.__class__.__name__} fit={len(train)} predict={len(test)}")
            # Print amount of data for each label in train
            X = train.to_numpy()
            for label in train.iloc[:, -1].unique():
                print(f"{label}={len(X[X[:,-1] == label])}")

        model.fit(train.to_numpy())
        predictions = model.predict(test.to_numpy())
        metrics.compute(predictions, verbose)

    print(f"\n{model_name_for_print} \033[1;34m# Final Metrics\033[0m")
    metrics.show_final_metrics()

