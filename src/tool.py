import logging

# local imports
from sampling.Sampler import Sampler
from training.data_loader import DataLoader
from membership_inference.membership_inference_kde import MembershipPredictorKDE
from membership_inference.kernel_density_estimation import KDE
from gaussian_analysis.gaussian_analysis import GaussianAnalysis
from util.plot_creator import plot_densities, plot_loss_arrays
from util import utils
from training.model import Model
from kl_divergence.kl_divergence import KLDivergence, KLDivergenceVisualizer
from constants import DATASET_PATH, RESULTS_PATH, EXPERIMENT_NAME, LOSS_TYPE


logging.getLogger().setLevel(logging.INFO)


def evaluate(percentage, model_name):
    logging.info(f"Started Evaluation For (percentage = {percentage})")

    # create the results folder
    utils.create_results_directory()

    # get known training dataset and private dataset.
    data_loader = DataLoader(path=DATASET_PATH)
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()

    # perform inference and compute the densities
    model = Model("../models", model_name)
    gaussian_analysis = GaussianAnalysis(model, utils.compute_loss, LOSS_TYPE)

    # get loss for each known and unknown image 
    logging.info("getting loss arrays for the entire dataset...")
    known_loss_array, unknown_loss_array = gaussian_analysis.get_loss_arrays(
        x_train, x_test, y_train, y_test
    )

    # TO DO: add a loop over percentage values
    sampler = Sampler()

    (
        train_known_idx,
        train_private_idx,
        eval_known_idx,
        eval_private_idx,
    ) = sampler.sample(percentage, eval_percentage=0.05)

    # extact train known and private data
    train_known_loss = known_loss_array[train_known_idx]
    train_private_loss = unknown_loss_array[train_private_idx]

    # extract test known and private data
    test_known_loss = known_loss_array[eval_known_idx]
    test_private_loss = unknown_loss_array[eval_private_idx]

    # plot histogram of loss arrays
    plot_loss_arrays(train_known_loss, train_private_loss)

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

    # compute metrics
    cf_train = utils.confusion_matrix(train_known_pred, train_private_pred)
    logging.info(f"train confusion matrix: {cf_train}")
    cf_test = utils.confusion_matrix(test_known_pred, test_private_pred)
    logging.info(f"test confusion matrix: {cf_test}")

    # ratios
    train_ratios = utils.to_rate(cf_train)
    test_ratios = utils.to_rate(cf_test)

    # log results
    utils.log_results(cf_train, cf_test, train_ratios, test_ratios)
    plot_densities(train_known_loss, train_private_loss)
    
    # compute the KL Divergence
    kl_divergence = KLDivergence()
    kl_divergence_value = kl_divergence.compute_discrete_single(train_known_loss, train_private_loss)
    print(f"The KL Divergence value is {round(kl_divergence_value, 4)}.")

if __name__ == "__main__":
    models = ["/baseline_lenet5_100_epochs.pth"]
    evaluate(5, models[0])
