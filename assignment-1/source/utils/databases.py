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
