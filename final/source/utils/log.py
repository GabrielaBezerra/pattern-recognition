verbose: bool = False


def model(model, database):
    if verbose:
        print(f"\n\033[1;33m# {model} Experiment in {database} \033[0m")


def database(database):
    if verbose:
        print(f"\n\033[1;32m# {database} database \033[0m")


def realization_details(i, model, split, train, test, metrics):
    if verbose:
        print(f"\n\033[1;31m{model.name}\033[0m \033[1;32m#{i} Realization\033[0m")
        print_split_details(split, train, test)
        print_metrics_for_realization(metrics)


def print_split_details(split, train, test):
    if verbose:
        print(f"split_method={split.__class__.__name__}")
        print(f"train={len(train)}")
        X = train.to_numpy()
        for label in sorted(train.iloc[:, -1].unique()):
            print(f"\t{label}={len(X[X[:,-1] == label])}")
        print(f"test={len(test)}")
        Y = test.to_numpy()
        for label in sorted(test.iloc[:, -1].unique()):
            print(f"\t{label}={len(Y[Y[:,-1] == label])}")


def print_metrics_for_realization(metrics):
    (confusion_matrix, hit_rates, std_dict) = metrics.summary
    if verbose:
        print("\n\033[1;34m# Metrics\033[0m")
        print("\nConfusion Matrix:")
        print(confusion_matrix)
        print("\nHit Rates:")
        for label, hit_rate in hit_rates.items():
            print(f"{label}: {hit_rate:.2f}")
        print("\nStandard Deviation:")
        for label, std in std_dict.items():
            print(f"{label}: {std:.2f}")


def experiment_results(final_metrics, database_name):
    print(f"\n\033[1;34m##### Final Metrics for {database_name}\033[0m")

    # generating markdown table
    for metric in final_metrics.keys():
        if metric == "Confusion Matrix":
            print("\n\033[1;30mConfusion Matrix\033[0m\n")
            for model, confusion_matrix in final_metrics[metric].items():
                print(f"\n\033[1;31m{model}\033[0m\n")
                print("| ", end="")
                for label in confusion_matrix.columns:
                    print(f"|{label}", end="")
                print("|")
                print("|-|" + ("-|" * len(confusion_matrix.columns)))
                for index, row in confusion_matrix.iterrows():
                    print(f"|{index}|{'|'.join(map(str, row))}|")
            continue
        labels = list(final_metrics[metric].items())[0][1].keys()
        print(f"\n\033[1;30m{metric}\033[0m\n")
        print("| ", end="")
        for label in labels:
            print(f"|{label}", end="")
        print("|")
        print("|-|" + ("-|" * len(labels)))
        for model, dict in final_metrics[metric].items():
            values = list(map(lambda v: f"{v}"[:4], dict.values()))
            print(f"|{model}|{"|".join(values)}|")

    # TODO: Should also show confusion matrix for average realizations / metrics
