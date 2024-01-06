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

