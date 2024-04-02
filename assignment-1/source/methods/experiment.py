import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import graphs
from .metrics import ClassifierMetrics

def realizations(df: pd.DataFrame, model, split, times=10, verbose=True):
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

        train, test = split.split(df)
        graphs.plot_database_after_split(
            f"Realization {i}", 
            train,
            test,
            feature_a=0,
            feature_b=1
        )

        if verbose:
            print(f"split_method={split.__class__.__name__} fit={len(train)} predict={len(test)}")
            # Print amount of data for each label in train
            X = train.to_numpy()
            for label in train.iloc[:, -1].unique():
                print(f"{label}={len(X[X[:,-1] == label])}")

        model.fit(train.to_numpy())
        predictions = model.predict(test.to_numpy())
        metrics.compute(predictions, verbose)
        
        # TODO: Plot the decision boundaries
        # x_min, x_max = train.iloc[:, 0].min(), train.iloc[:, 0].max() + 1
        # y_min, y_max = train.iloc[:, 1].min(), train.iloc[:, 1].max() + 1
        # xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        # Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

        # # Put the result into a color plot
        # Z = Z[0]
        # plt.figure()
        # plt.pcolormesh(xx, yy, Z, cmap="viridis")

        # # Plot also the training points
        # plt.scatter(X[:, 0], X[:, 1], edgecolors='k', cmap="viridis")
        # plt.xlim(xx.min(), xx.max())
        # plt.ylim(yy.min(), yy.max())
        # plt.title(f"Realization {i}")

        # plt.show()

    print(f"\n{model_name_for_print} \033[1;34m# Final Metrics\033[0m")
    metrics.show_final_metrics()

