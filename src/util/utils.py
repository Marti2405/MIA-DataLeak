import os
import json
import shutil
import logging
import numpy as np
from util.plot_creator import plot_confusion_matrix
from constants import RESULTS_PATH, EXPERIMENT_NAME, EPSILON


def create_results_directory():
    logging.info("create results directory...")

    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    if os.path.exists(RESULTS_PATH + EXPERIMENT_NAME):
        shutil.rmtree(RESULTS_PATH + EXPERIMENT_NAME)

    os.makedirs(RESULTS_PATH + EXPERIMENT_NAME)

def confusion_matrix(pred_know: np.ndarray, pred_private: np.ndarray) -> tuple:
    tp = np.count_nonzero(pred_know == 1)
    fn = np.count_nonzero(pred_know == 0)

    fp = np.count_nonzero(pred_private == 1)
    tn = np.count_nonzero(pred_private == 0)

    return tp, fp, tn, fn

def to_rate(confusion_matrix: tuple) -> tuple:
    tp, fp, tn, fn = confusion_matrix

    # Compute True positive rate...
    fpr = fp / (fp + tn) * 100
    tpr = tp / (tp + fn) * 100
    fnr = fn / (fn + tp) * 100
    tnr = tn / (tn + fp) * 100

    return tpr, fpr, tnr, fnr

def log_results(cf_train, cf_test, train_ratios, test_ratios) -> None:
    metrics_dict = to_metrics_dict(cf_train, cf_test, train_ratios, test_ratios)

    # raw metrics
    with open(f"{RESULTS_PATH}{EXPERIMENT_NAME}/data.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=4)

    # plot
    plot_confusion_matrix(cf_train, "train")
    plot_confusion_matrix(cf_test, "test")

def to_dict(cf: tuple, ratios: tuple) -> dict:
    return {
        "confusion_matrix": {
            "tp": cf[0],
            "fp": cf[1],
            "tn": cf[2],
            "fn": cf[3],
        },
        "ratios": {
            "tpr": ratios[0],
            "fpr": ratios[1],
            "tnr": ratios[2],
            "fnr": ratios[3],
        },
    }

def to_metrics_dict(cf_train, cf_test, train_ratios, test_ratios) -> dict:
    train_dict = to_dict(cf_train, train_ratios)
    test_dict = to_dict(cf_test, test_ratios)
    return {"train": train_dict, "test": test_dict}
