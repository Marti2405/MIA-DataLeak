import logging
import numpy as np
from tqdm import tqdm

# local imports
from sampling.Sampler import Sampler
from training.data_loader import DataLoader
from membership_inference.membership_inference_kde import MembershipPredictorKDE
from membership_inference.kernel_density_estimation import KDE
from loss_calculator.loss_calculator import LossCalculator
from util.plot_creator import plot_densities, plot_loss_arrays, plot_kl_divergence, plot_confusion_matrix
from util import utils
from training.model import Model
from kl_divergence.kl_divergence import KLDivergence
from constants import DATASET_PATH, LOSS_TYPE, PERCENTAGES, MODEL_NAME, TRIALS


logging.getLogger().setLevel(logging.INFO)


def run_experiment_percentage(percentage, known_loss_array, unknown_loss_array):
    logging.info(f"Started evaluation for {percentage}% of data")

    # initialize variables that accumulate various metrics
    cf_train_sum = np.array((0, 0, 0, 0))
    cf_test_sum = np.array((0, 0, 0, 0))
    train_ratios_sum = np.array((0, 0, 0, 0))
    test_ratios_sum = np.array((0, 0, 0, 0))
    kl_divergence_sum = 0

    for _ in tqdm(range(TRIALS)):
        # sample data indices
        (
            train_known_idx,
            train_private_idx,
            eval_known_idx,
            eval_private_idx,
        ) = Sampler().sample(percentage)

        # extact train known and private data
        train_known_loss = known_loss_array[train_known_idx]
        train_private_loss = unknown_loss_array[train_private_idx]

        # extract test known and private data
        test_known_loss = known_loss_array[eval_known_idx]
        test_private_loss = unknown_loss_array[eval_private_idx]

        # plot histogram of loss arrays
        plot_loss_arrays(train_known_loss, train_private_loss, percentage)

        # approximate the known and unknown densities using the KDE
        known_density = KDE(train_known_loss)
        private_density = KDE(train_private_loss)

        # instantiate the membership predictor class
        membership_predictor_kde = MembershipPredictorKDE(known_density, private_density)

        # predict training:
        train_known_pred = membership_predictor_kde.predict(
            train_known_loss
        )
        train_private_pred = membership_predictor_kde.predict(
            train_private_loss
        )

        # predict test:
        test_known_pred = membership_predictor_kde.predict(
            test_known_loss
        )
        test_private_pred = membership_predictor_kde.predict(
            test_private_loss
        )

        # plot train densities estimated with KDE
        plot_densities(train_known_loss, train_private_loss, percentage, "training")

        # plot train densities estimated with KDE
        plot_densities(test_known_loss, test_private_loss, percentage, "testing")

        # compute and add the KL Divergence value
        kl_divergence_sum += KLDivergence().compute_discrete_single(train_known_loss, train_private_loss)

        # compute metrics and ratios and sum them
        cf_train = utils.confusion_matrix(train_known_pred, train_private_pred)
        cf_test = utils.confusion_matrix(test_known_pred, test_private_pred)
        cf_train_sum = np.add(cf_train_sum, cf_train)
        cf_test_sum = np.add(cf_test_sum, cf_test)
        train_ratios_sum = np.add(train_ratios_sum, utils.to_rate(cf_train))
        test_ratios_sum = np.add(test_ratios_sum, utils.to_rate(cf_test))

    # compute the average of the cumulated metrics and KL Divergence
    cf_train_mean = tuple(cf_train_sum / TRIALS)
    cf_test_mean = tuple(cf_test_sum / TRIALS)
    train_ratios_mean = tuple(train_ratios_sum / TRIALS)
    test_ratios_mean = tuple(test_ratios_sum / TRIALS)
    kl_divergence_mean = kl_divergence_sum / TRIALS

    # log results
    utils.log_results(cf_train_mean, cf_test_mean, train_ratios_mean, test_ratios_mean, percentage)

    # plot confusion matrices for train and test data
    plot_confusion_matrix(cf_train_mean, "trainining", percentage)
    plot_confusion_matrix(cf_test_mean, "test", percentage)

    return kl_divergence_mean

def run_experiment():
    logging.info(f"Loss type: {LOSS_TYPE}")

    # create the results folder
    utils.create_results_directory()

    # get known training dataset and private dataset.
    data_loader = DataLoader(path=DATASET_PATH)
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()

    # perform inference and compute the densities
    model = Model("../models", MODEL_NAME)
    loss_calculator = LossCalculator(model, LOSS_TYPE)

    # get loss for each known and unknown image 
    logging.info("Getting loss arrays for the entire dataset...")
    known_loss_array, unknown_loss_array = loss_calculator.get_loss_arrays(
        x_train, x_test, y_train, y_test
    )

    kl_divergence_values = []

    for percentage in PERCENTAGES:
        kl_divergence_mean = run_experiment_percentage(percentage, known_loss_array, unknown_loss_array)
        kl_divergence_values.append(kl_divergence_mean)

    plot_kl_divergence(PERCENTAGES, kl_divergence_values)

if __name__ == "__main__":
    run_experiment()
