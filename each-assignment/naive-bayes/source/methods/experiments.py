from utils import databases, log
from utils.plot import PlotFactory
from methods.preprocessing import Preprocessing
from methods.metrics import ClassifierMetrics


class Experiment:
    def __init__(self, database_name, df, plot) -> None:
        self.database_name = database_name
        self.df = df
        self.classes = {}
        self.plot = plot
        self.metrics = []

    def realizations(
        self,
        models,
        split,
        times,
        plots,
        plot_delay=1.0,
    ):
        # Preprocessing
        preprocessing = Preprocessing(self.df)
        self.df = (
            preprocessing.removing_rows_with_empty_features()
            .transforming_columns_to_numerical()
            .removing_identical_rows()
            .preprocessed_dataframe
        )
        if self.classes == {}:
            self.classes = preprocessing.classes
        self.plot.classes = self.classes

        # Realizations
        for realiz in range(1, times + 1):
            # Split
            train, test = split.split(self.df)

            # For each model
            for model in models:
                log.model(model.name, self.database_name)

                # Train
                model.fit(train.to_numpy())

                # Test
                predictions = model.predict(test.to_numpy())

                # Metrics
                realiz_met = ClassifierMetrics(model.name, realiz, split, predictions)
                self.metrics.append(realiz_met)
                log.realization_details(realiz, model, split, train, test, realiz_met)

                # Plots
                self.plot.show(plots, model, realiz, train, test, plot_delay)

        # TODO: compute final metrics for model from all realizations
        final_metrics = ClassifierMetrics.compute_final_metrics(
            self.metrics, self.classes
        )
        log.experiment_results(final_metrics, self.database_name)


main = [
    Experiment(
        database_name="Artificial I",
        df=databases.loadArtificial(n=1),
        plot=PlotFactory(
            database_name="Artificial I (0,1)",
            features=(0, 1),
            decision_boundary_step=0.05,
        ),
    ),
    Experiment(
        database_name="Artificial II",
        df=databases.loadArtificial(n=2),
        plot=PlotFactory(
            database_name="Artificial II (0,1)",
            features=(0, 1),
            decision_boundary_step=0.05,
        ),
    ),
    Experiment(
        database_name="Iris",
        df=databases.loadIris(),
        plot=PlotFactory(
            database_name="Iris (2,3)",
            features=(2, 3),
            decision_boundary_step=0.05,
        ),
    ),
    Experiment(
        database_name="Column 2D",
        df=databases.loadColumn(binary=True),
        plot=PlotFactory(
            database_name="Column 2D (1,5)",
            features=(1, 5),
            decision_boundary_step=0.5,
        ),
    ),
    Experiment(
        database_name="Column 3D",
        df=databases.loadColumn(binary=False),
        plot=PlotFactory(
            database_name="Column 3D (3,4)",
            features=(3, 4),
            decision_boundary_step=0.5,
        ),
    ),
    Experiment(
        database_name="Dermatology",
        df=databases.loadDermatology(),
        plot=PlotFactory(
            database_name="Dermatology (1,2)",
            features=(1, 2),
            decision_boundary_step=0.05,
        ),
    ),
    Experiment(
        database_name="Breast Cancer",
        df=databases.loadBreastCancer(),
        plot=PlotFactory(
            database_name="Breast Cancer (1,9)",
            features=(1, 9),
            decision_boundary_step=0.05,
        ),
    ),
]


def create_permutations(arr):
    return [(arr[i], arr[j]) for i in range(len(arr)) for j in range(i + 1, len(arr))]


artificial_1 = [
    Experiment(
        database_name="Artificial I",
        df=databases.loadArtificial(n=1),
        plot=PlotFactory(
            database_name="Artificial I",
            features=(0, 1),
            decision_boundary_step=0.05,
        ),
    ),
]

artificial_2 = [
    Experiment(
        database_name="Artificial II",
        df=databases.loadArtificial(n=2),
        plot=PlotFactory(
            database_name="Artificial II",
            features=(0, 1),
            decision_boundary_step=0.05,
        ),
    ),
]

iris = [
    Experiment(
        database_name="Iris",
        df=databases.loadIris(),
        plot=PlotFactory(
            database_name=f"Iris {permutation}",
            features=permutation,
            decision_boundary_step=0.1,
        ),
    )
    for permutation in create_permutations(range(4))
]

column_2d = [
    Experiment(
        database_name="Column 2D",
        df=databases.loadColumn(binary=True),
        plot=PlotFactory(
            database_name=f"Column 2D {permutation}",
            features=permutation,
            decision_boundary_step=0.5,
        ),
    )
    for permutation in create_permutations(range(6))
]

column_3d = [
    Experiment(
        database_name="Column 3D",
        df=databases.loadColumn(binary=False),
        plot=PlotFactory(
            database_name=f"Column 3D {permutation}",
            features=permutation,
            decision_boundary_step=0.5,
        ),
    )
    for permutation in create_permutations(range(6))
]

dermatology = [
    Experiment(
        database_name="Dermatology",
        df=databases.loadDermatology(),
        plot=PlotFactory(
            database_name=f"Dermatology {permutation}",
            features=permutation,
            decision_boundary_step=0.05,
        ),
    )
    for permutation in create_permutations(range(34))
]

breast_cancer = [
    Experiment(
        database_name="Breast Cancer",
        df=databases.loadBreastCancer(),
        plot=PlotFactory(
            database_name=f"Breast Cancer {permutation}",
            features=permutation,
            decision_boundary_step=0.05,
        ),
    )
    for permutation in create_permutations(range(10))
]

additional = (
    artificial_1
    + artificial_2
    + iris
    + column_2d
    + column_3d
    + dermatology
    + breast_cancer
)

all = main + additional
