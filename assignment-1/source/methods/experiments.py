import pandas as pd
import numpy as np


def realizations(df: pd.DataFrame, model, split_method, times=10):
    """
    Perform multiple realizations of a pattern recognition experiment.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        model: The pattern recognition model to be used.
        split_method: The method for splitting the dataset into training and testing sets.
        times (int, optional): The number of realizations to perform. Defaults to 10.
    """

    hit_rates = {}

    # Realizations loop
    for i in range(1, times + 1):
        print("\n# Realization", i)

        train, test = split_method.split(df)
        model.fit(train.to_numpy())
        predictions = model.predict(test.to_numpy())

        # show confusion matrix for realization using crosstab without skipping indexes
        confusion_matrix = confusion_matrix = pd.crosstab(
            [prediction[1] for prediction in predictions], # true values
            [prediction[0][-1] for prediction in predictions], # predicted values
            rownames=["True"], colnames=["Predicted"],
            dropna=False
        )
        print(f"\nConfusion Matrix - Realization {i} \n{confusion_matrix}\n")

        # Compute true and false values for each class from confusion matrix
        true_positives = confusion_matrix.values.diagonal()
        false_negatives = (confusion_matrix.sum(axis=1) - true_positives).to_numpy()

        # Compute hit or miss for each class
        for i, label in enumerate(confusion_matrix.index):
            hit_rate = true_positives[i] / (true_positives[i] + false_negatives[i])
            print(f"Hit rate for {label}: {hit_rate:.2f}")
            hit_rates[label] = hit_rates.get(label, []) + [hit_rate]

    print("\n# Final Metrics")
    # Compute average hit rate for each class
    for label, hit_rates_list in hit_rates.items():
        avg_hit_rate = np.average(hit_rates_list)
        print(f"Acurracy for {label} (average hit rate for {label}): {avg_hit_rate:.2f}")

    # Compute overall average hit rate
    overall_hit_rate = np.average([hit_rate for hit_rates_list in hit_rates.values() for hit_rate in hit_rates_list]) # flatten the dictionary
    print(f"Overall Accuracy (overall average hit rate): {overall_hit_rate:.2f}")

    # Compute standard deviation for each class
    std_rates = {}
    for label, hit_rates_list in hit_rates.items():
        std_rate = np.std(hit_rates_list)
        std_rates[label] = std_rate
        print(f"Standard deviation for {label}: {std_rate:.2f}")
    
    # Compute overall standard deviation
    overall_std_rate = np.std([hit_rate for hit_rates_list in hit_rates.values() for hit_rate in hit_rates_list]) # flatten the dictionary
    print(f"Overall Standard Deviation: {overall_std_rate:.2f}")

