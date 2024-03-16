import numpy as np


class ClassifierMetric:
    def __init__(self, label: str):
        self.label = label
        self.hit_rates = []
        self.std_rates = []
        self._accuracy = 0
        self._std = 0

    @property
    def accuracy(self):
        np.average(self.hit_rates)

    @property
    def std(self):
        np.std(self.std_rates)

    def compute(self, hit_miss: list[int]):
        self.hit_rates.append(np.average(hit_miss))
        self.std_rates.append(np.std(hit_miss))
