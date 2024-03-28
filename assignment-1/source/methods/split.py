import pandas as pd


def holdout(df: pd.DataFrame, train_percent: float = 0.5):
    df = df.sample(frac=1, replace=False)
    train_count = int(len(df) * train_percent)
    test_count = len(df) - train_count
    train = df[:train_count]
    test = df[-test_count:]
    return train, test
