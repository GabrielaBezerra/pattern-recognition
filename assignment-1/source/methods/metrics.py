import numpy as np
import pandas as pd


class ClassifierMetrics:
    def __init__(self):
        self.all_hit_rates = {}

    def confusion_matrix(self, predictions: list[tuple[np.ndarray, str]]):
        return pd.crosstab(
            [prediction[1] for prediction in predictions],  # true values
            [prediction[0][-1] for prediction in predictions],  # predicted values
            rownames=["True"],
            colnames=["Predicted"],
            dropna=False,
        )

    def compute(self, predictions):
        hit_miss_realization = {}
        hit_rates_realization = {"All": 0.0}
        std_dict_realization = {"All": 0.0}

        confusion_matrix = self.confusion_matrix(predictions)
        true_positives = confusion_matrix.values.diagonal()
        false_negatives = (confusion_matrix.sum(axis=1) - true_positives).to_numpy()

        all_positives = np.sum(true_positives)
        all_negatives = np.sum(false_negatives)
        all_hit_rate_realization = all_positives / (all_positives + all_negatives)
        hit_rates_realization["All"] = all_hit_rate_realization
        self.all_hit_rates["All"] = self.all_hit_rates.get("All", []) + [
            all_hit_rate_realization
        ]

        for i, label in enumerate(confusion_matrix.index):
            # Compute Hit Rate
            hit_rate = true_positives[i] / (true_positives[i] + false_negatives[i])
            hit_rates_realization[label] = hit_rate
            self.all_hit_rates[label] = self.all_hit_rates.get(label, []) + [hit_rate]

            # Compute Hit Miss for Standard Deviation
            hit_miss_realization[label] = []
            for _ in range(0, true_positives[i]):
                hit_miss_realization[label].append(1)
            for _ in range(0, false_negatives[i]):
                hit_miss_realization[label].append(0)

        std_dict_realization["All"] = float(
            np.std([item for list in hit_miss_realization.values() for item in list])
        )
        for label in hit_miss_realization.keys():
            std_dict_realization[label] = float(np.std(hit_miss_realization[label]))

        return (confusion_matrix, hit_rates_realization, std_dict_realization)

    def compute_final_metrics(self):
        final_accuracies = {}
        final_std = {}

        # Compute average hit rate for each class
        for label, hit_rates_list in self.all_hit_rates.items():
            avg_hit_rate = np.average(hit_rates_list)
            final_accuracies[label] = avg_hit_rate

        # Compute standard deviation from hit rates for each class
        for label, hit_rates_list in self.all_hit_rates.items():
            std = np.std(hit_rates_list)
            final_std[label] = std

        return {"Accuracy": final_accuracies, "Standard Deviation": final_std}
