import numpy as np
import pandas as pd

df = pd.read_csv("datasets/iris/iris.csv")

#       Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm         Species
# 0      1            5.1           3.5            1.4           0.2     Iris-setosa
# 1      2            4.9           3.0            1.4           0.2     Iris-setosa
# 2      3            4.7           3.2            1.3           0.2     Iris-setosa
# 3      4            4.6           3.1            1.5           0.2     Iris-setosa
# 4      5            5.0           3.6            1.4           0.2     Iris-setosa
# ..   ...            ...           ...            ...           ...             ...
# 145  146            6.7           3.0            5.2           2.3  Iris-virginica
# 146  147            6.3           2.5            5.0           1.9  Iris-virginica
# 147  148            6.5           3.0            5.2           2.0  Iris-virginica
# 148  149            6.2           3.4            5.4           2.3  Iris-virginica
# 149  150            5.9           3.0            5.1           1.8  Iris-virginica

# [150 rows x 6 columns]

def split(df: pd.DataFrame, train_percent: float = 0.7):
    df = df.sample(frac=1)[:10]
    train_count = int(len(df) * train_percent)
    test_count = len(df) - train_count
    train = df[:train_count]
    test = df[-test_count:]
    return (train, test) # TODO: split in (train, train_labels, test, test_labels)

all_avg_versicolor = []
all_avg_virginica = []
all_avg_setosa = []

all_std_versicolor = []
all_std_virginica = []
all_std_setosa = []

for r in range(1,11):
    # train, train_labels, test, test_labels = split(df)
    
    # knn.fit(train, train_labels)
    
    # res = knn.predict(test, test_labels)
    
    # metrics
    versicolor = [1,0,0]
    virginica = [1,0,1]
    setosa = [1,1,1]
    
    avg_versicolor = np.average(versicolor)
    avg_virginica = np.average(virginica)
    avg_setosa = np.average(setosa)
    all_avg_versicolor.append(avg_versicolor)
    all_avg_virginica.append(avg_virginica)
    all_avg_setosa.append(avg_setosa)

    std_versicolor = np.std(versicolor)
    std_virginica = np.std(virginica)
    std_setosa = np.std(setosa)
    all_std_versicolor.append(std_versicolor)
    all_std_virginica.append(std_virginica)
    all_std_setosa.append(std_setosa)
