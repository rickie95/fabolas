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
    path = f"./results/{folder}"

    if not os.path.exists(path):
        os.makedirs(path)

    pd.DataFrame(results).to_csv(os.path.join(path, f"results_{method}_{datetime.datetime.now()}"))
    pd.DataFrame(progress).to_csv(os.path.join(path, f"progress_{method}_{datetime.datetime.now()}"))
    logging.info(f"Results saved in {path}.")
