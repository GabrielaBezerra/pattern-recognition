from zmq import has
from .metrics import ClassifierMetrics
import numpy as np
import matplotlib.pyplot as plt


def realizations(df, model, split, plot=None, times=10, verbose=True):
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
        # if plot:
        #     plot.show_database_after_split(
        #         f"{model.name} - Realization {i}", train, test
        #     )

        if verbose:
            print(
                f"split_method={split.__class__.__name__} fit={len(train)} predict={len(test)}"
            )
            # Print amount of data for each label in train
            X = train.to_numpy()
            for label in train.iloc[:, -1].unique():
                print(f"{label}={len(X[X[:,-1] == label])}")

        model.fit(train.to_numpy())
        predictions = model.predict(test.to_numpy())
        metrics.compute(predictions, verbose)

        # TODO: Plot the decision boundaries
        h = 0.5
        feat_a = 0
        feat_b = 1
        x_min, x_max = (
            train.iloc[:, feat_a].min() - 0.25,
            train.iloc[:, feat_a].max() + 0.25,
        )
        y_min, y_max = (
            train.iloc[:, feat_b].min() - 0.25,
            train.iloc[:, feat_b].max() + 0.25,
        )
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        if train.shape[1] > 3:
            model.fit(train.iloc[:, [feat_a, feat_b, -1]].to_numpy())
        else:
            model.fit(train.to_numpy())
        tuples_list = model.predict(np.c_[xx.ravel(), yy.ravel()], has_labels=False)
        Z_list = [t[1] for t in tuples_list]
        num_labels = {}
        categorical_label = False
        if isinstance(Z_list[0], str):
            categorical_label = True
            for i, label in enumerate(train.iloc[:, -1].unique()):
                num_labels[label] = i
                num_labels[i] = label
            Z_num = [num_labels[z] for z in Z_list]
        else:
            Z_num = Z_list
        Z = np.array(Z_num).reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8)

        if isinstance(train.iloc[0, -1], str):
            colors = [num_labels[label] for label in train.iloc[:, -1]]
        else:
            colors = train.iloc[:, -1]
        scatter = plt.scatter(
            train.iloc[:, feat_a],
            train.iloc[:, feat_b],
            c=colors,
            edgecolors="k",
        )
        plt.xlabel(train.columns[feat_a])
        plt.ylabel(train.columns[feat_b])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(f"{model.name} - Decision Boundary - Realization {i}")
        # show all legends in the plot from num_labels
        if categorical_label:
            labels = [
                f"{num_labels[int(''.join(i for i in value if i.isdigit()))]} ({value})"
                for value in scatter.legend_elements()[1]
            ]
        else:
            labels = scatter.legend_elements()[1]
        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=labels,
            title=train.columns[-1],
        )
        # else:
        #     plt.legend(
        #         handles=scatter.legend_elements()[0],
        #         labels=scatter.legend_elements()[1],
        #         title=train.columns[-1],
        #     )
        plt.tight_layout()
        plt.show(block=True)
        plt.close()

    print(f"\n{model_name_for_print} \033[1;34m# Final Metrics\033[0m")
    metrics.show_final_metrics()
