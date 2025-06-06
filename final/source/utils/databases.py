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


def loadDermatology():
    return pd.read_csv(
        "datasets/dermatology/dermatology.data",
        names=[
            "erythema",
            "scaling",
            "definite borders",
            "itching",
            "koebner phenomenon",
            "polygonal papules",
            "follicular papules",
            "oral mucosal involvement",
            "knee and elbow involvement",
            "scalp involvement",
            "family history",
            "melanin incontinence",
            "eosinophils in the infiltrate",
            "PNL infiltrate",
            "fibrosis of the papillary dermis",
            "exocytosis",
            "acanthosis",
            "hyperkeratosis",
            "parakeratosis",
            "clubbing of the rete ridges",
            "elongation of the rete ridges",
            "thinning of the suprapapillary epidermis",
            "spongiform pustule",
            "munro microabcess",
            "focal hypergranulosis",
            "disappearance of the granular layer",
            "vacuolisation and damage of basal layer",
            "spongiosis",
            "saw-tooth appearance of retes",
            "follicular horn plug",
            "perifollicular parakeratosis",
            "inflammatory monoluclear inflitrate",
            "band-like infiltrate",
            "Age",
            "Class",
        ],
        delimiter=",",
    )


def loadBreastCancer():
    return pd.read_csv(
        "datasets/breast-cancer/breast-cancer.data",
        names=[
            "Class",
            "age",
            "menopause",
            "tumor-size",
            "inv-nodes",
            "node-caps",
            "deg-malig",
            "breast",
            "breast-quad",
            "irradiat",
        ],
        delimiter=",",
    )

def loadPairProgrammingSocialStyles():
    return pd.read_csv(
        "datasets/pair-programming/pair_programming_social_styles.csv",
        header=0,
        delimiter=",",
    )