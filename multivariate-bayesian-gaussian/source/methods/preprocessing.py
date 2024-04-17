import pandas as pd
import numpy as np


class Preprocessing:
    def __init__(self, df):
        self.df = df
        self.classes = {}

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
        if self.classes == {}:
            if (
                self.df.iloc[:, -1].dtype == "O"
                and not self.df.iloc[:, -1].str.match(r"^\d+$").all()
            ):
                self.classes = {
                    code: value
                    for code, value in enumerate(self.df.iloc[:, -1].unique())
                }
            else:
                self.classes = {
                    value: value
                    for code, value in enumerate(self.df.iloc[:, -1].unique())
                }
        for i in range(len(self.df.columns)):
            if self.df.iloc[:, i].dtype == "O":
                # check if the column is a string with exclusively number characters
                if self.df.iloc[:, i].str.match(r"^\d+$").all():
                    self.df.iloc[:, i] = pd.to_numeric(self.df.iloc[:, i])
                else:
                    self.df.iloc[:, i] = pd.Categorical(self.df.iloc[:, i]).codes
                self.df = self.df.astype({self.df.columns[i]: "float64"})
        return self
