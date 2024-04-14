import numpy as np
import matplotlib.pyplot as plt
import warnings


class Plot:
    def __init__(
        self,
        database_name: str,
        features: tuple[int, int],
        decision_boundary_step=0.1,
    ):
        self.database_name = database_name
        self.feature_a = features[0]
        self.feature_b = features[1]
        self.decision_boundary_step = decision_boundary_step

    def show_database_after_split(self, model, r, train, test, delay=0):
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
        ax.set(xlim=(x_min - 0.25, x_max + 0.25), ylim=(y_min - 0.25, y_max + 0.25))
        ax.set(xlabel=feat_name_a, ylabel=feat_name_b)
        plt.title(f"Split - {model.name} - {self.database_name} - R{r}")
        plt.grid(True)
        plt.legend(
            title=train.columns[-1],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            shadow=True,
            ncol=2,
        )
        plt.tight_layout()
        plt.tight_layout()
        if delay > 0:
            plt.show(block=False)
            plt.pause(delay)
            plt.close()
        else:
            plt.show(block=True)

    def show_decision_boundary(self, model, r, df, delay=0):
        feat_a = self.feature_a
        feat_b = self.feature_b
        step = self.decision_boundary_step

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
        num_labels = {}
        categorical_label = False
        if isinstance(Z_list[0], str):
            categorical_label = True
            for i, label in enumerate(df.iloc[:, -1].unique()):
                num_labels[label] = i
                num_labels[i] = label
            Z_num = [num_labels[z] for z in Z_list]
        else:
            Z_num = Z_list
        Z = np.array(Z_num).reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8)

        if isinstance(df.iloc[0, -1], str):
            colors = [num_labels[label] for label in df.iloc[:, -1]]
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
        plt.title(f"Decision Boundary - {model.name} - {self.database_name} - R{r}")
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
            title=f"Train {df.columns[-1]}",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            shadow=True,
            ncol=2,
        )
        plt.tight_layout()
        if delay > 0:
            plt.show(block=False)
            plt.pause(delay)
            plt.close()
        else:
            plt.show(block=True)

    def show_gaussian_3d(self, model, r, df, delay=0):
        feat_a = self.feature_a
        feat_b = self.feature_b
        step = self.decision_boundary_step

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
        num_labels = {}
        categorical_label = False
        if isinstance(Z_list[0], str):
            categorical_label = True
            for i, label in enumerate(df.iloc[:, -1].unique()):
                num_labels[label] = i
                num_labels[i] = label
            Z_num = [num_labels[z] for z in Z_list]
        else:
            Z_num = Z_list
        Z = np.array(Z_num).reshape(xx.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xx, yy, Z, alpha=0.8)
        ax.set_xlabel(df.columns[feat_a])
        ax.set_ylabel(df.columns[feat_b])
        ax.set_zlabel("Class")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_title(f"Gaussian 3D - {model.name} - {self.database_name} - R{r}")
        # show all legends in the plot from num_labels
        if delay > 0:
            plt.show(block=False)
            plt.pause(delay)
            plt.close()
        else:
            plt.show(block=True)
