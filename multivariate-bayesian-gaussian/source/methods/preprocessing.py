import pandas as pd


class Preprocessing:
    def __init__(self, df):
        self.df = df

    @property
    def preprocessed_dataframe(self):
        return self.df

    def removing_rows_with_empty_features(self):
        self.df = self.df.dropna()
        return self

    def transforming_columns_to_numerical(self):
        for i in range(len(self.df.columns)):
            if self.df.iloc[:, i].dtype == "O":
                self.df.iloc[:, i] = pd.Categorical(self.df.iloc[:, i]).codes
                self.df = self.df.astype({self.df.columns[i]: "float64"})
        return self
