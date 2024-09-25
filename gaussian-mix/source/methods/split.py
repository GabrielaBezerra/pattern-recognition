import pandas as pd


class Holdout:
    def __init__(self, train_percent: float = 0.5):
        self.train_percent = train_percent

    def split(self, df: pd.DataFrame):
        df = df.sample(frac=1, replace=False)
        train_count = int(len(df) * self.train_percent)
        test_count = len(df) - train_count
        train = df[:train_count]
        test = df[-test_count:]
        return train, test
