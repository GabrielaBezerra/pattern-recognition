import os
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import warnings
from models.bayesian_gaussian_multivariate import BayesianGaussianMultivariate
from models.naive_bayes import NaiveBayesClassifier


class Plot:
    TRAIN_TEST = "train_test"
    DECISION_BOUNDARY = "decision_boundary"
    DECISION_BOUNDARY_3D = "decision_boundary_3d"
    GAUSSIAN_CURVES_3D = "gaussian_curves_3d"


class PlotFactory:
    def __init__(
        self,
        database_name: str,
        features: tuple[int, int],
        decision_boundary_step=0.5,
    ):
        self.database_name = database_name
        self.feature_a = features[0]
        self.feature_b = features[1]
        self.decision_boundary_step = decision_boundary_step
        self.classes = {}

    def show(self, plots, model, r, train, test, delay=0):
        self.plots = plots
        self.model = model
        self.r = r
        self.train = train
        self.test = test
        self.delay = delay

        if Plot.TRAIN_TEST in plots:
            plt = self.show_database_after_split()
            self.save_plot(plt, Plot.TRAIN_TEST)

        if Plot.DECISION_BOUNDARY in plots:
            plt = self.show_decision_boundary()
            self.save_plot(plt, Plot.DECISION_BOUNDARY)

        if Plot.DECISION_BOUNDARY_3D in plots:
            plt = self.show_decision_boundary_3d()
            self.save_plot(plt, Plot.DECISION_BOUNDARY_3D)

        if Plot.GAUSSIAN_CURVES_3D in plots:
            plt = self.show_gaussian_curves_3d()

    def show_database_after_split(self):
        # Copy properties to variables
        feature_a = self.feature_a
        feature_b = self.feature_b
        train = type(self.train)(self.train)
        test = type(self.test)(self.test)

        # Check if the features are categorical and replace them with numbers
        warnings.filterwarnings("ignore")
        categorical_dict = {}
        if isinstance(train.iloc[0, feature_a], str):
            for n, label in enumerate(train.iloc[:, feature_a].unique()):
                train.iloc[:, feature_a] = train.iloc[:, feature_a].replace(label, n)
                test.iloc[:, feature_a] = test.iloc[:, feature_a].replace(label, n)
                categorical_dict[n] = label
        if isinstance(train.iloc[0, feature_b], str):
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
                label=(
                    f"Train {self.classes[i]} ({label})"
                    if "Dermatology" not in self.database_name
                    else f"Train {label}"
                ),
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
        ax.set(xlim=(x_min - 0.25, x_max + 0.25), ylim=(y_min - 0.25, y_max + 0.25))
        ax.set(xlabel=feat_name_a, ylabel=feat_name_b)
        plt.title(f"Split - {self.model.name} - {self.database_name} - R{self.r}")
        plt.grid(True)
        plt.legend(
            title=train.columns[-1],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            shadow=True,
            ncol=2,
        )
        plt.tight_layout()
        self.save_plot(plt, Plot.TRAIN_TEST)
        self.show_with_delay(plt)

    def show_decision_boundary(self):
        feat_a = self.feature_a
        feat_b = self.feature_b
        step = self.decision_boundary_step
        model = copy(self.model)
        df = self.train

        x_min, x_max = (
            df.iloc[:, feat_a].min() - 0.25,
            df.iloc[:, feat_a].max() + 0.25,
        )
        y_min, y_max = (
            df.iloc[:, feat_b].min() - 0.25,
            df.iloc[:, feat_b].max() + 0.25,
        )
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, step), np.arange(y_min, y_max, step)
        )
        if df.shape[1] > 3:
            model.fit(df.iloc[:, [feat_a, feat_b, -1]].to_numpy())
        else:
            model.fit(df.to_numpy())
        tuples_list = model.predict(np.c_[xx.ravel(), yy.ravel()], has_labels=False)
        Z_list = [t[1] for t in tuples_list]
        num_labels = self.classes
        if isinstance(Z_list[0], str):
            for i, label in enumerate(self.classes):
                num_labels[self.classes[i]] = i
            Z_num = [num_labels[z] for z in Z_list]
        else:
            Z_num = Z_list
        Z = np.array(Z_num).reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8)

        if isinstance(df.iloc[0, -1], str):
            colors = [num_labels[label] for label in self.classes]
        else:
            colors = df.iloc[:, -1]
        scatter = plt.scatter(
            df.iloc[:, feat_a],
            df.iloc[:, feat_b],
            c=colors,
            edgecolors="k",
        )
        plt.xlabel(df.columns[feat_a])
        plt.ylabel(df.columns[feat_b])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(
            f"Decision Boundary - {self.model.name} - {self.database_name} - R{self.r}"
        )
        # show all legends in the plot from num_labels
        labels = [self.classes[i] for i in num_labels]
        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=labels,
            title=f"Train {df.columns[-1]}",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            shadow=True,
            ncol=2,
        )
        plt.tight_layout()
        self.save_plot(plt, Plot.DECISION_BOUNDARY)
        self.show_with_delay(plt)

    def show_decision_boundary_3d(self):
        feat_a = self.feature_a
        feat_b = self.feature_b
        step = self.decision_boundary_step
        model = copy(self.model)
        df = self.train

        x_min, x_max = (
            df.iloc[:, feat_a].min() - 0.25,
            df.iloc[:, feat_a].max() + 0.25,
        )
        y_min, y_max = (
            df.iloc[:, feat_b].min() - 0.25,
            df.iloc[:, feat_b].max() + 0.25,
        )
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, step), np.arange(y_min, y_max, step)
        )
        if df.shape[1] > 3:
            model.fit(df.iloc[:, [feat_a, feat_b, -1]].to_numpy())
        else:
            model.fit(df.to_numpy())
        tuples_list = model.predict(np.c_[xx.ravel(), yy.ravel()], has_labels=False)
        Z_list = [t[1] for t in tuples_list]
        Z = np.array(Z_list).reshape(xx.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xx, yy, Z, alpha=0.8)  # type: ignore
        ax.set_xlabel(df.columns[feat_a])
        ax.set_ylabel(df.columns[feat_b])
        ax.set_zlabel(df.columns[-1])  # type: ignore
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_title(
            f"Decision Boundary 3D - {self.model.name} - {self.database_name} - R{self.r}"
        )
        plt.tight_layout()
        self.save_plot(plt, Plot.DECISION_BOUNDARY_3D)
        self.show_with_delay(plt)

    def show_gaussian_curves_3d(self):
        if (
            type(self.model) is not BayesianGaussianMultivariate
            and type(self.model) is not NaiveBayesClassifier
        ):
            warnings.warn(
                "This plot is only available for BayesianGaussianMultivariate and NaiveBayesClassifier models."
            )
            return

        feat_a = self.feature_a
        feat_b = self.feature_b
        step = self.decision_boundary_step
        bottom_value = 0.00005
        model = copy(self.model)
        df = self.train

        x_min, x_max = (
            df.iloc[:, feat_a].min() - 0.25,
            df.iloc[:, feat_a].max() + 0.25,
        )
        y_min, y_max = (
            df.iloc[:, feat_b].min() - 0.25,
            df.iloc[:, feat_b].max() + 0.25,
        )
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, step), np.arange(y_min, y_max, step)
        )
        if df.shape[1] > 3:
            model.fit(df.iloc[:, [feat_a, feat_b, -1]].to_numpy())
        else:
            model.fit(df.to_numpy())
        # plot all classes density probability functions at the same time
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for i, _ in enumerate(model.classes):
            x = np.linspace(x_min, x_max, 100)
            y = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros(X.shape)
            for j in range(X.shape[0]):
                for k in range(X.shape[1]):
                    if type(model) is BayesianGaussianMultivariate:
                        Z[j, k] = model._multivariate_gaussian(
                            np.array([X[j, k], Y[j, k]]),
                            model.class_means[i],
                            model.class_covs[i],
                        )
                    elif type(model) is NaiveBayesClassifier:
                        Z[j, k] = model._univariate_gaussian(
                            np.array([X[j, k], Y[j, k]]),
                            model.class_means[i],
                            model.class_vars[i],
                        )
                    else:
                        print("none")

            Z[Z <= bottom_value] = np.nan
            ax.plot_surface(X, Y, Z, alpha=0.8)  # type: ignore
        ax.set_xlabel(df.columns[feat_a])
        ax.set_ylabel(df.columns[feat_b])
        ax.set_zlabel("Density Probability")  # type: ignore
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_title(f"Gaussian 3D - {model.name} - {self.database_name} - R{self.r}")
        labels = [
            f"{self.classes[int(i)]}"
            if "Dermatology" not in self.database_name
            else f"{c}"
            for i, c in enumerate(model.classes)
        ]
        plt.legend(
            labels,
            title="Classes",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            shadow=True,
            ncol=2,
        )
        plt.tight_layout()
        self.save_plot(plt, Plot.GAUSSIAN_CURVES_3D)
        self.show_with_delay(plt)

    def show_with_delay(self, plt):
        if self.delay > 0:
            plt.show(block=False)
            plt.pause(self.delay)
            plt.close()
        else:
            plt.show(block=True)

    # Save the plot after showing it
    def save_plot(self, plt, Plot):
        if plt is None:
            return
        plot_filename = f"plots/{self.database_name}_{self.model.name}_plot_{Plot}.png"
        os.makedirs('plots', exist_ok=True)
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
