import pandas as pd


def loadArtificial(n=1):
    return pd.read_csv(f"datasets/artificial/artificial-{n}.csv")


def loadColumn(binary=True):
    return pd.read_csv(
        f"datasets/coluna/column_{2 if binary else 3}C.dat",
        names=[
            "PelvicIncidence",
            "PelvicTilt",
            "LumbarLordosisAngle",
            "SacralSlope",
            "PelvicRadius",
            "DegreeOfSpondylolisthesis",
            "Class",
        ],
        delimiter=" ",
    )


def loadIris():
    """
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
    """
    return pd.read_csv(
        "datasets/iris/iris.data",
        names=[
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
            "Species",
        ],
    )
