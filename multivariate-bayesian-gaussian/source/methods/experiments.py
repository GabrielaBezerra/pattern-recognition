from utils import databases, log
from utils.plot import PlotFactory, Plot
from methods.preprocessing import Preprocessing
from models.bayesian_gaussian_multivariate import BayesianGaussianMultivariate


class Experiment:
    def __init__(self, database_name, df, plot) -> None:
        self.database_name = database_name
        self.df = df
        self.classes = {}
        self.plot = plot

    def realizations(
        self,
        model,
        split,
        times,
        metrics,
        plots,
        plot_delay=1.0,
    ):
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

        for r in range(1, times + 1):
            train, test = split.split(self.df)

            if Plot.TRAIN_TEST in plots:
                self.plot.show_database_after_split(
                    model,
                    r,
                    train,
                    test,
                    delay=plot_delay,
                )

            model.fit(train.to_numpy())
            predictions = model.predict(test.to_numpy())
            realization_metrics = metrics.compute(predictions)
            log.realization_details(r, model, split, train, test, realization_metrics)

            if Plot.DECISION_BOUNDARY in plots:
                self.plot.show_decision_boundary(
                    model,
                    r,
                    train,
                    delay=plot_delay,
                )

            if Plot.DECISION_BOUNDARY_3D in plots:
                self.plot.show_decision_boundary_3d(
                    model,
                    r,
                    train,
                    delay=plot_delay,
                )

            if (
                type(model) is BayesianGaussianMultivariate
                and Plot.GAUSSIAN_CURVES_3D in plots
            ):
                self.plot.show_gaussian_curves_3d(
                    model,
                    r,
                    train,
                    delay=plot_delay,
                )

        final_metrics = metrics.compute_final_metrics(self.classes)

        log.experiment_results(final_metrics)


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

aditional = (
    artificial_1
    + artificial_2
    + iris
    + column_2d
    + column_3d
    + dermatology
    + breast_cancer
)

all = main + aditional
