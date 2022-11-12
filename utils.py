import os
import datetime
import logging
import pandas as pd


def save_results(results, progress, method, folder):
    # Check if path exist:
    # ./results/svm_mnist
    # if not create it
    # then save two files
    # results_{method}_{datetime}.csv
    # progress_{method}_{datetime}.csv
    path = f"./results/{folder}/{method}"

    if not os.path.exists(path):
        os.makedirs(path)

    results_df = pd.DataFrame()
    progress_df = pd.DataFrame()

    # Split X and config in multiple columns if needed
    for col_index in range(results["X"].shape[1]):
        results_df[f"X_{col_index}"] = results["X"][:, col_index]

    for col_index in range(progress["config"].shape[1]):
        progress_df[f"config_{col_index}"] = progress["config"][:, col_index]

    results_df["y"] = results["y"]
    progress_df["value"] = progress["value"]

    progress_df["time"] = progress["time"]

    if method == "fabolas":
        results_df["size"] = results["size"]
        results_df["c"] = results["c"]
        progress_df["size"] = progress["size"]

    results_df.to_csv(os.path.join(path, f"results_{method}_{datetime.datetime.now()}"))
    progress_df.to_csv(os.path.join(path, f"progress_{method}_{datetime.datetime.now()}"))
    logging.info(f"Results saved in {path}.")


def print_usage(args):
    logging.error(f"Call this script as: {args[0]} [method]")
    logging.error("Where 'method' can be one of the following: "
                  "'random_search' 'expected_improvement'"
                  "'entropy_search' 'fabolas'.")
    logging.error("Default is 'random_search'")
