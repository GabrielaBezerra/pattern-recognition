import numpy as np


class ClassifierMetric:
    def __init__(self, label: str):
        self.label = label
        self.hit_rates = []
        self.std_rates = []
        self.accuracy = 0
        self.std = 0

    def compute(self, hit_miss: list[int]):
        self.hit_rates.append(np.average(hit_miss))
        self.std_rates.append(np.std(hit_miss))
        self.accuracy = np.average(self.hit_rates)
        self.std = np.std(self.std_rates)

    def log(self):
        print(
            f"\nLabel: {self.label}\nAccuracy: {self.accuracy:.2f}\nStandard Deviation: {self.std:.2f}"
        )