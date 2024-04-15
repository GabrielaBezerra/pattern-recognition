import pandas as pd
import numpy as np


class Preprocessing:
    def __init__(self, df):
        self.df = df

    @property
    def preprocessed_dataframe(self):
        return self.df

    def removing_rows_with_empty_features(self):
        self.df = self.df.replace("?", np.nan).dropna()
        return self

    def removing_features_with_missing_values(self):
        self.df = self.df.replace("?", np.nan).dropna(axis=1)
        return self

    def removing_identical_rows(self):
        self.df = self.df.drop_duplicates()
        return self

    def transforming_columns_to_numerical(self):
        for i in range(len(self.df.columns)):
            if self.df.iloc[:, i].dtype == "O":
                self.df.iloc[:, i] = pd.to_numeric(self.df.iloc[:, i])
                # self.df.iloc[:, i] = pd.Categorical(self.df.iloc[:, i]).codes / 10
                self.df = self.df.astype({self.df.columns[i]: "int64"})
        return self
