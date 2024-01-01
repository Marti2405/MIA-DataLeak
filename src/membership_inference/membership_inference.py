from Gaussian import Gaussian
import numpy as np


def predict(
    known_gaussian: Gaussian, unknown_gaussian: Gaussian, sample: np.ndarray
) -> float:
    """returns ratio, higher means higher likelihood of being part of training dataset. 1 means equal."""
    loss = get_loss(sample)
    p_from_known_gaussian = known_gaussian.pdf(loss)
    p_from_unknown_gaussian = unknown_gaussian.pdf(loss)
    return p_from_known_gaussian / p_from_unknown_gaussian


def get_loss(sample: np.ndarray):
    return 4


if __name__ == "__main__":
    training_gaussian = Gaussian(0, 1)
    private_gaussian = Gaussian(5, 1)
    sample = np.zeros((34, 34, 3))
    print(predict(training_gaussian, private_gaussian, sample))
