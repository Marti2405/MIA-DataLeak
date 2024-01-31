from training.model import Model
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

EPSILON = 1e-7

class GaussianAnalysis:
    model: Model

    def __init__(self, model, compute_loss, loss_type):
        self.model = model
        # function that computes the type of chosen loss
        self.compute_loss = compute_loss
        self.loss_type = loss_type

    def get_loss_arrays(self, known, unknown, test_known, test_unknown):
        """
        Compute the loss array for two data samples
        Input: Two Samples from Data - Known & Unknown (for both of them - train and test)
        Output: Loss Array for each sample - known_loss_array, unknown_loss_array
        """
        known_prob = self.model.predict(known)
        unknown_prob = self.model.predict(unknown)

        predictions_prob_known = np.array([
            prob[test_known[i][0]] for i, prob in enumerate(known_prob)
        ])

        predictions_prob_unknown = np.array([
            prob[test_unknown[i][0]] for i, prob in enumerate(unknown_prob)
        ])

        loss_known = self.compute_loss(predictions_prob_known, self.loss_type)
        loss_unknown = self.compute_loss(predictions_prob_unknown, self.loss_type)

        return loss_known, loss_unknown

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
        if not self.check_normal_distribution(data):
            raise Exception("Not normally distributed!")

        mean_value = np.mean(data)
        std_deviation = np.std(data)

        return mean_value, std_deviation
