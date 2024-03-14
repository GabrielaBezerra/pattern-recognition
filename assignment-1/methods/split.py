import pandas as pd

def holdout(df: pd.DataFrame, train_percent: float = 0.7):
    df = df.sample(frac=1)
    train_count = int(len(df) * train_percent)
    test_count = len(df) - train_count
    train = df[:train_count]
    test = df[-test_count:]
    train_features = train.iloc[:, :-1]
    train_labels = train.iloc[:, -1]
    test_features = test.iloc[:, :-1]
    test_labels = test.iloc[:, :-1]
    return (train_features, train_labels, test_features, test_labels) # TODO: split in (train, train_labels, test, test_labels)
