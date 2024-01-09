import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

IMAGE_PATH = "../../data/"


class KLDivergence:
    """
    This class computes the KL Divergence between
    pairs of continuous or discrete distributions.
    """

    epsilon = 1e-32

    def __compute_continuous(self, mu_p, sigma_p, mu_q, sigma_q):
        """
        This auxiliary method computes the KL Divergence between 2 continuous
        Gaussian distributions.
        """

        # if the std. dev. is zero, it adds an epsilon value to avoid computation errors
        if sigma_p == 0.0:
            sigma_p += self.epsilon

        if sigma_q == 0.0:
            sigma_q += self.epsilon

        kl_divergence = (
            np.log(sigma_q / sigma_p)
            + (sigma_p**2 + (mu_p - mu_q) ** 2) / (2 * sigma_q**2)
            - 0.5
        )

        return kl_divergence

    def compute_continuous(self, mus_p, sigmas_p, mus_q, sigmas_q):
        """
        It computes the KL Divergence between any number of pairs of Gaussian
        distributions defined as lists of means and standard deviations.
        """

        # treat the case when only a pair of distributions has been passed
        if type(mus_p) != list and type(mus_p) != np.ndarray:
            return [self.__compute_continuous(mus_p, sigmas_p, mus_q, sigmas_q)]

        # test if all input lists have the same length
        condition = len(mus_p) == len(sigmas_p) == len(mus_q) == len(sigmas_q)

        assert condition, "Must pass parameters with the same number of values!"

        kl_divergences = []

        # for each pair compute and store the KL Divergence
        for mu_p, sigma_p, mu_q, sigma_q in zip(mus_p, sigmas_p, mus_q, sigmas_q):
            kl_divergences.append(
                self.__compute_continuous(mu_p, sigma_p, mu_q, sigma_q)
            )

        return kl_divergences

    def __compute_discrete(self, distrib_p, distrib_q):
        """
        This auxiliary method computes the KL Divergence between 2 discrete
        distributions of any type passed as lists.
        """

        # check if all values are greater or equal to zero
        assert (
            min(distrib_p) >= 0.0 and min(distrib_q) >= 0.0
        ), "Probability values should be positive!"

        distrib_p_length = len(distrib_p)
        distrib_q_length = len(distrib_q)

        # check if the input lists have the same number of elements
        assert (
            distrib_p_length == distrib_q_length
        ), "Distributions must have the same length!"

        kl_divergence = 0

        # compute the KL Divergence
        for index in range(distrib_p_length):
            # if value is zero, it adds an epsilon to avoid computation errors
            if distrib_p[index] == 0.0:
                distrib_p[index] += self.epsilon

            if distrib_q[index] == 0.0:
                distrib_q[index] += self.epsilon
            
            kl_divergence += distrib_p[index] * np.log(
                distrib_p[index] / distrib_q[index]
            )

        return kl_divergence

    def compute_discrete(self, distribs_p, distribs_q):
        """
        It computes the KL Divergence between any number of pairs of 
        distributions of any type defined as lists of values.
        """

        # treat the case when only a pair of distributions has been passed
        if type(distribs_p[0]) != list and type(distribs_p[0]) != np.ndarray:
            return [self.__compute_discrete(distribs_p, distribs_q)]

        # check if each list contains the same number of distributions
        distrib_p_no = len(distribs_p)
        distrib_q_no = len(distribs_q)

        assert (
            distrib_p_no == distrib_q_no
        ), "Must pass the same number of distributions!"

        kl_divergences = []

        # compute the KL Divergence for each pair of input distributions
        for distrib_p, distrib_q in zip(distribs_p, distribs_q):
            kl_divergences.append(self.__compute_discrete(distrib_p, distrib_q))

        return kl_divergences


class KLDivergenceVisualizer:
    """
    This class plots and stores the KL Divergence values
    as a function of percentage of leaked training data.
    """

    def __init__(self):
        self.default_title = "KL Divergence between the probability\ndistributions of leaked training and testing data"
        self.default_img_title = "kl_divergence"

    def plot(
        self,
        percentages,
        kl_divergence_values,
        img_title=None,
        plot_title=None,
        save_figure=False,
    ):
        """
        Plot and store a scatter plot of the KL Divergence values.
        """

        # if titles haven't been passed, use the default ones
        if plot_title is None:
            plot_title = self.default_title

        if img_title is None:
            img_title = self.default_img_title

        # create the scatter plot
        plt.plot(percentages, kl_divergence_values, marker="o")
        plt.title(plot_title)
        plt.xlabel("Percentage of disclosed training data")
        plt.ylabel("KL Divergence")

        # save the plot as a PNG image
        if save_figure:
            plt.savefig(IMAGE_PATH + img_title + ".png", bbox_inches="tight")

        # display the created figure
        plt.show()
