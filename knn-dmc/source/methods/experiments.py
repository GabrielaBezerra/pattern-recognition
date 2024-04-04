from utils import databases
from utils.plot import Plot


class Experiment:
    def __init__(self, database_name, df, plot) -> None:
        self.database_name = database_name
        self.df = df
        self.plot = plot

    def realizations(self, model, split, times, metrics, log):
        for r in range(1, times + 1):
            train, test = split.split(self.df)

            self.plot.show_database_after_split(
                model,
                r,
                train,
                test,
                delay=1,
            )

            model.fit(train.to_numpy())
            predictions = model.predict(test.to_numpy())
            realization_metrics = metrics.compute(predictions)
            log.realization_details(r, model, split, train, test, realization_metrics)

            self.plot.show_decision_boundary(
                model,
                r,
                train,
                delay=1,
            )

        final_metrics = metrics.compute_final_metrics()

        log.experiment_results(final_metrics)


main = [
    Experiment(
        database_name="Artificial I",
        df=databases.loadArtificial(),
        plot=Plot(
            database_name="Artificial I",
            features=(0, 1),
            decision_boundary_step=0.05,
        ),
    ),
    Experiment(
        database_name="Iris",
        df=databases.loadIris(),
        plot=Plot(
            database_name="Iris (2,3)",
            features=(2, 3),
            decision_boundary_step=0.1,
        ),
    ),
    Experiment(
        database_name="Column 2D",
        df=databases.loadColumn(binary=True),
        plot=Plot(
            database_name="Column 2D (0,1)",
            features=(4, 5),
            decision_boundary_step=0.5,
        ),
    ),
    Experiment(
        database_name="Column 3D",
        df=databases.loadColumn(binary=False),
        plot=Plot(
            database_name="Column 3D (0,1)",
            features=(4, 5),
            decision_boundary_step=0.5,
        ),
    ),
]


def create_permutations(arr):
    return [(arr[i], arr[j]) for i in range(len(arr)) for j in range(i + 1, len(arr))]


artificial = [
    Experiment(
        database_name="Artificial I",
        df=databases.loadArtificial(),
        plot=Plot(
            database_name="Artificial I",
            features=(0, 1),
            decision_boundary_step=0.05,
        ),
    ),
]

iris = [
    Experiment(
        database_name="Iris",
        df=databases.loadIris(),
        plot=Plot(
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
        plot=Plot(
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
        plot=Plot(
            database_name=f"Column 3D {permutation}",
            features=permutation,
            decision_boundary_step=0.5,
        ),
    )
    for permutation in create_permutations(range(6))
]

aditional = artificial + iris + column_2d + column_3d

all = main + aditional
