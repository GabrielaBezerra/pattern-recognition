import matplotlib.pyplot as plt
import warnings


class Plot:
    def __init__(self, feature_a: int, feature_b: int, delay=1):
        self.feature_a = feature_a
        self.feature_b = feature_b
        self.delay = delay

    def show_database_after_split(self, title, train, test):
        # Copy properties to variables
        feature_a = self.feature_a
        feature_b = self.feature_b

        # Check if the features are categorical and replace them with numbers
        warnings.filterwarnings("ignore")
        categorical_dict = {}
        categorical_label = False
        if isinstance(train.iloc[0, feature_a], str):
            categorical_label = feature_a == -1
            for n, label in enumerate(train.iloc[:, feature_a].unique()):
                train.iloc[:, feature_a] = train.iloc[:, feature_a].replace(label, n)
                test.iloc[:, feature_a] = test.iloc[:, feature_a].replace(label, n)
                categorical_dict[n] = label
        if isinstance(train.iloc[0, feature_b], str):
            categorical_label = feature_b == -1
            for n, label in enumerate(train.iloc[:, feature_b].unique()):
                train.iloc[:, feature_b] = train.iloc[:, feature_b].replace(label, n)
                test.iloc[:, feature_b] = test.iloc[:, feature_b].replace(label, n)
                categorical_dict[n] = label
        warnings.filterwarnings("default")

        # Plot the database
        plt.rcParams["figure.figsize"] = [5, 5]
        fig, ax = plt.subplots()
        for i, label in enumerate(train.iloc[:, -1].unique()):
            ax.scatter(
                train[train.iloc[:, -1] == label].iloc[:, feature_a],
                train[train.iloc[:, -1] == label].iloc[:, feature_b],
                c=[f"C{i}"],
                label=f"Train {label}"
                + (f" ({categorical_dict[label]})" if categorical_label else ""),
            )
        ax.scatter(
            test.iloc[:, feature_a],
            test.iloc[:, feature_b],
            c="black",
            marker="*",
            label="Test",
        )
        feat_name_a = train.columns[feature_a]
        feat_name_b = train.columns[feature_b]
        x_min = train.iloc[:, feature_a].min()
        x_max = train.iloc[:, feature_a].max()
        y_min = train.iloc[:, feature_b].min()
        y_max = train.iloc[:, feature_b].max()
        ax.set(xlim=(x_min - 1, x_max + 1), ylim=(y_min - 1, y_max + 1))
        ax.set(xlabel=feat_name_a, ylabel=feat_name_b)
        plt.title(title)
        plt.grid(True)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        plt.tight_layout()
        plt.show(block=False)
        if self.delay > 0:
            plt.pause(self.delay)
        plt.close()
