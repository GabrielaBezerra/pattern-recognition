import numpy as np
import pandas as pd


class ClassifierMetrics:
    def __init__(self, model_name, r, split, predictions):
        self.all_hit_rates = {}
        self.model_name = model_name
        self.realization_number = r
        self.split = split
        self.predictions = predictions
        self._compute()

    def confusion_matrix(self):
        m = pd.crosstab(
            [prediction[1] for prediction in self.predictions],  # predicted values
            [prediction[0][-1] for prediction in self.predictions],  # true values
            rownames=["Predicted"],
            colnames=["True"],
            dropna=False,
        )
        # fill empty classes with zeros to make m symmetric
        for label in set(m.index).symmetric_difference(set(m.columns)):
            if label not in m.columns:
                m[label] = 0
            if label not in m.index:
                m.loc[label] = 0
        m = m.sort_index(axis=0).sort_index(axis=1)
        return m

    def _compute(self):
        hit_miss_realization = {}
        hit_rates_realization = {"All": 0.0}
        std_dict_realization = {"All": 0.0}

        confusion_matrix = self.confusion_matrix()
        true_positives = confusion_matrix.values.diagonal()
        false_negatives = (confusion_matrix.sum(axis=0) - true_positives).to_numpy()

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

        self.summary = (confusion_matrix, hit_rates_realization, std_dict_realization)

    @staticmethod
    def compute_final_metrics(metrics: list["ClassifierMetrics"], classes):
        if len(metrics) == 0:
            exit("No metrics to compute final metrics")

        final_accuracies = {}
        final_std = {}

        # list all possible models from metrics
        models = set([m.model_name for m in metrics])

        # list all possible labels from metrics
        labels = metrics[0].all_hit_rates.keys()

        # Compute average and standard deviation of hit rate for each class, by model
        for model in models:
            for label in labels:
                hit_rates_per_label_model = [
                    m.all_hit_rates[label]
                    for m in metrics
                    if m.model_name == model
                    if m.all_hit_rates.get(label) is not None
                ]
                # convert label to class from classes
                class_name = classes.get(label, label)
                accuracy = np.average(hit_rates_per_label_model)
                standard_deviation = np.std(hit_rates_per_label_model)
                if final_accuracies.get(model) is None:
                    final_accuracies[model] = {class_name: accuracy}
                else:
                    final_accuracies[model][class_name] = accuracy
                if final_std.get(model) is None:
                    final_std[model] = {class_name: standard_deviation}
                else:
                    final_std[model][class_name] = standard_deviation
        
        closest_to_average_case_confusion_matrix = {model: None}
        for model in models:
            closest_diff = 1
            summaries_per_model = [m.summary for m in metrics if m.model_name == model]
            for summary in summaries_per_model:
                if final_accuracies[model]["All"] == summary[1]["All"]:
                    closest_to_average_case_confusion_matrix[model] = summary[0]
                    break
                diff = abs(final_accuracies[model]["All"] - summary[1]["All"])
                if diff < closest_diff:
                    closest_diff = diff
                    closest_to_average_case_confusion_matrix[model] = summary[0]
                    
        return {"Accuracy": final_accuracies, "Standard Deviation": final_std, "Confusion Matrix": closest_to_average_case_confusion_matrix}
