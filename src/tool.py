import numpy as np
import logging
import os
import shutil
import seaborn as sn
import json
import matplotlib.pyplot as plt

# local imports
from sampling.Sampler import Sampler
from training.data_loader import DataLoader
from membership_inference.membership_inference import MembershipPredictor
from membership_inference.Gaussian import Gaussian
from gaussian_analysis.gaussian_analysis import GaussianAnalysis
from training.model import Model
from constants import EPSILON


logging.getLogger().setLevel(logging.INFO)

# define the type of loss
LOSS_TYPE = "normalized_probability"

# define results folder path
RESULTS_PATH = "../results/"
DATASET_PATH = "../data/raw_dataset/"
EXPERIMENT_NAME = f'lenet_100_epochs_{LOSS_TYPE}_loss'


def compute_loss(predictions_prob, loss_type):
    print("Loss type: ", loss_type)
    if loss_type == "probability":
        return np.array([1 - prob for prob in predictions_prob])
    elif loss_type == "cross_entropy":
        return np.array([-np.log(prob) for prob in predictions_prob])
    elif loss_type == "normalized_probability":
        return np.log(predictions_prob / (1 - predictions_prob + EPSILON))
    else:
        raise Exception("This type of loss it not defined.")

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


def save_confusion_matrix(cf, name):
    # Create a 2D array from the cf tuple
    cf_array = np.array([[cf[0], cf[3]], [cf[1], cf[2]]])

    # Create a seaborn heatmap
    sn.heatmap(
        cf_array,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=["True", "False"],
        yticklabels=["True", "False"],
    )

    # Set axis labels and title
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")

    # Save the plot
    plt.savefig(f"{RESULTS_PATH}{EXPERIMENT_NAME}/confusion_matrix_{name}.jpg", dpi=300)
    plt.show()


def log_results(cf_train, cf_test, train_ratios, test_ratios) -> None:
    metrics_dict = to_metrics_dict(cf_train, cf_test, train_ratios, test_ratios)
    i = os.listdir(RESULTS_PATH)

    # raw metrics
    with open(f"{RESULTS_PATH}{EXPERIMENT_NAME}/data.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=4)

    # plot
    save_confusion_matrix(cf_train, "train")
    save_confusion_matrix(cf_test, "test")


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


def log_gaussian_plot(gaussian_known: Gaussian, gaussian_private: Gaussian):
    gaussian_known.compare(gaussian_private)
    plt.savefig(f"{RESULTS_PATH}{EXPERIMENT_NAME}/gaussian.jpg", dpi=300)
    plt.show()


def evaluate(percentage, model_name):
    logging.info(f"Started Evaluation For (percentage = {percentage})")

    # create the results folder
    create_results_directory()

    # get known training dataset and private dataset.
    data_loader = DataLoader(path=DATASET_PATH)
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()

    sampler = Sampler()

    (
        train_known_idx,
        train_private_idx,
        eval_known_idx,
        eval_private_idx,
    ) = sampler.sample(percentage, eval_percentage=0.05)

    train_known_x = x_train[train_known_idx]
    train_private_x = x_test[train_private_idx]
    train_known_y = y_train[train_known_idx]
    train_private_y = y_test[train_private_idx]

    test_known_x = x_train[eval_known_idx]
    test_private_x = x_test[eval_private_idx]
    test_known_y = y_train[eval_known_idx]
    test_private_y = y_test[eval_private_idx]

    # perform inference and compute the gaussians
    model = Model("../models", model_name)
    gaussian_analysis = GaussianAnalysis(model, compute_loss, LOSS_TYPE)
    membership_predictor = MembershipPredictor(model, compute_loss, LOSS_TYPE)

    logging.info("getting loss arrays...")

    # perform inference on training data.
    known_loss_array, unknown_loss_array = gaussian_analysis.get_loss_arrays(
        train_known_x, train_private_x, train_known_y, train_private_y
    )
    fig = gaussian_analysis.plot_loss_arrays(known_loss_array, unknown_loss_array)
    fig.savefig(f"{RESULTS_PATH}{EXPERIMENT_NAME}/losses.jpg", dpi=300)

    # get normal dist parameters.
    known_mean, known_std = gaussian_analysis.compute_mean_and_std(known_loss_array)
    private_mean, private_std = gaussian_analysis.compute_mean_and_std(
        unknown_loss_array
    )

    known_gaussian = Gaussian(known_mean, known_std)
    logging.info(f"Found know gaussian: {known_gaussian}")
    private_gaussian = Gaussian(private_mean, private_std)
    logging.info(f"Found private gaussian: {private_gaussian}")

    # predict training:
    train_known_classifications = membership_predictor.predict(
        known_gaussian, private_gaussian, train_known_x, train_known_y
    )
    train_private_classifications = membership_predictor.predict(
        known_gaussian, private_gaussian, train_private_x, train_private_y
    )

    # predict test:
    test_known_classifications = membership_predictor.predict(
        known_gaussian, private_gaussian, test_known_x, test_known_y
    )
    test_private_classifications = membership_predictor.predict(
        known_gaussian, private_gaussian, test_private_x, test_private_y
    )

    # compute metrics
    cf_train = confusion_matrix(
        train_known_classifications, train_private_classifications
    )
    logging.info(f"train confusion matrix: {cf_train}")
    cf_test = confusion_matrix(test_known_classifications, test_private_classifications)
    logging.info(f"test confusion matrix: {cf_train}")

    # ratios
    train_ratios = to_rate(cf_train)
    test_ratios = to_rate(cf_test)

    # log results.
    log_results(cf_train, cf_test, train_ratios, test_ratios)
    log_gaussian_plot(known_gaussian, private_gaussian)


if __name__ == "__main__":
    # understand the code
    # bring the modifications made in the other version (kde, CE computation)
    models = [
        "/baseline_resnet.pth",
        "/baseline_resnet_3_epochs.pth",
        "/baseline_lenet.pth",
    ]
    evaluate(5, models[2])
