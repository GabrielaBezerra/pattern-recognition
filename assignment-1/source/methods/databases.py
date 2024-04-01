import pandas as pd


def loadIris():
    '''
    Load the Iris dataset from the datasets folder.
    The dataset has the following columns:
    - SepalLengthCm
    - SepalWidthCm
    - PetalLengthCm
    - PetalWidthCm
    - Species (Iris-setosa, Iris-versicolor, Iris-virginica)
    The dataset has 150 rows.
    Example:
    >>> df = loadIris()
    >>> print(df)
    '''
    df = pd.read_csv(
        "datasets/iris/iris.data",
        names=[
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
            "Species",
        ],
    )
    return df
