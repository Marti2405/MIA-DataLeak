from training.model import Model
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


class GaussianAnalysis:
    model: Model

    def __init__(self, model):
        self.model = model

    def get_loss_arrays(self, known, unknown, test_known, test_unknown):
        """
        Compute the loss array for two data samples
        Input: Two Samples from Data - Known & Unknown (for both of them - train and test)
        Output: Loss Array for each sample - known_loss_array, unknown_loss_array
        """
        known_prob = self.model.predict(known)
        unknown_prob = self.model.predict(unknown)

        known_loss_array = self.model.get_loss(known_prob, test_known)
        unknown_loss_array = self.model.get_loss(unknown_prob, test_unknown)

        return known_loss_array, unknown_loss_array

    def plot_loss_arrays(self, arr1, arr2):
        """
        Plot both loss arrays - histograms
        Input: Loss arrays for Known/ Unknown Datasets
        Output: Plot of both losses
        """
        plt.figure(figsize=(10, 6))

        _, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

        axs[0].hist(arr1, bins=20)
        axs[1].hist(arr2, bins=20)

    def check_normal_distribution(self, data):
        """
        Check if a loss array follow the gaussian distribution
        Input: Loss array
        Output: Boolan value - Does the function follow a Gaussian Distribution?
        """
        _, p_value_shapiro = stats.shapiro(data)
        _, p_value_ks = stats.kstest(data, stats.norm.cdf)
        alpha = 0.05
        if p_value_shapiro < alpha and p_value_ks < alpha:
            return True
        else:
            return False

    def compute_mean_and_std(self, data):
        """
        Compute the mean and the standard deviation of a loss array
        Precondition: data should be a normal distribution;
        Input: Loss array
        Output: mean, standard deviation
        """
        print(data.shape)
        if not self.check_normal_distribution(data):
            # raise Exception("Not normally distributed!")
            print("warning")

        mean_value = np.mean(data)
        std_deviation = np.std(data)

        print(np.sum(mean_value))
        print("mean: ", mean_value)
        print("std: ", std_deviation)

        return mean_value, std_deviation
