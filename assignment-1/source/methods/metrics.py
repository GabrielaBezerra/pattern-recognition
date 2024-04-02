import numpy as np
import pandas as pd


class ClassifierMetrics:
    def __init__(self):
        self.hit_rates = {}
        self.std_rates = {}

    def confusion_matrix(self, predictions: list[tuple[np.ndarray, str]]):
        return pd.crosstab(
            [prediction[1] for prediction in predictions], # true values
            [prediction[0][-1] for prediction in predictions], # predicted values
            rownames=["True"], colnames=["Predicted"],
            dropna=False
        )

    def compute(self, predictions, verbose):
        confusion_matrix = self.confusion_matrix(predictions)
        if verbose: 
            print(f"\nConfusion Matrix: \n{confusion_matrix}\n")

        # Compute true and false values for each class from confusion matrix
        true_positives = confusion_matrix.values.diagonal()
        false_negatives = (confusion_matrix.sum(axis=1) - true_positives).to_numpy()

        # For each class
        for i, label in enumerate(confusion_matrix.index):
            # Compute Hit Rate
            hit_rate = true_positives[i] / (true_positives[i] + false_negatives[i])
            if verbose: 
                print(f"Hit rate for {label}: {hit_rate:.2f}")
            self.hit_rates[label] = self.hit_rates.get(label, []) + [hit_rate]

            # Compute Standard Deviation
            hit_miss = []
            for _ in range(0,true_positives[i]):
                hit_miss.append(1)
            for _ in range(0,false_negatives[i]):
                hit_miss.append(0)
            std = np.std(hit_miss)
            self.std_rates[label] = self.std_rates.get(label, []) + [std]
            if verbose:
                print(f"Std rate for {label}: {std:.2f}")

    def show_final_metrics(self):

        # Compute overall average hit rate
        overall_hit_rate = np.average([hit_rate for hit_rates_list in self.hit_rates.values() for hit_rate in hit_rates_list]) # flatten the dictionary
        print(f"\nOverall Accuracy:  {overall_hit_rate:.2f}")

        # Compute average hit rate for each class
        for label, hit_rates_list in self.hit_rates.items():
            avg_hit_rate = np.average(hit_rates_list)
            print(f"Acurracy for {label}: {avg_hit_rate:.2f}")

        # Compute overall standard deviation from hit_rates
        # overall_std_rate = np.std([hit_rate for hit_rates_list in self.hit_rates.values() for hit_rate in hit_rates_list]) # flatten the dictionary
        # print(f"\nOverall Standard Deviation:  {overall_std_rate:.2f}")

        # Compute standard deviation for each class
        # std of hit_rates
        # for label, hit_rates_list in self.hit_rates.items():
        #     std_rate = np.std(hit_rates_list)
        #     self.std_rates[label] = std_rate
        #     print(f"Standard deviation for {label}: {std_rate:.2f}")

        # Compute overall standard deviation from all std_rates
        overall_std_rate = np.std([std_rate for std_rates_list in self.std_rates.values() for std_rate in std_rates_list]) # flatten the dictionary
        print(f"\nOverall Standard Deviation:  {overall_std_rate:.2f}")

        # Compute standard deviation for each class
        # std of std_rates
        for label, std_rate_list in self.std_rates.items():
            std_rate = np.std(std_rate_list)
            print(f"Standard deviation for {label}: {std_rate:.2f}")
        