from .Gaussian import Gaussian
import numpy as np
import sys
import torch
from training.model import Model


class MembershipPredictor:
    model: Model

    def __init__(self, model):
        self.model = model

    def _predict(
        self,
        known_gaussian: Gaussian,
        unknown_gaussian: Gaussian,
        samples: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """returns ratio, higher means higher likelihood of being part of training dataset. 1 means equal."""
        loss = self.get_loss(samples, y)
        p_from_known_gaussian = np.array([known_gaussian.pdf(x) for x in loss])
        p_from_unknown_gaussian = np.array([unknown_gaussian.pdf(x) for x in loss])
        return p_from_known_gaussian / p_from_unknown_gaussian

    def predict(
        self,
        known_gaussian: Gaussian,
        unknown_gaussian: Gaussian,
        samples: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """true if part of original training set"""
        result = self._predict(known_gaussian, unknown_gaussian, samples, y)
        return (result > 1).astype(int)

    def get_loss(self, sample: np.ndarray, y: np.ndarray):
        print(sample.shape, y.shape)
        model_output = self.model.predict(sample)
        return self.model.get_loss(model_output, y)


if __name__ == "__main__":
    training_gaussian = Gaussian(0, 1)
    private_gaussian = Gaussian(5, 1)
    sample = np.zeros((1, 3, 34, 34)).astype(np.float32)
    # print(predict(training_gaussian, private_gaussian, sample))
    mp = MembershipPredictor()
    mp.predict(training_gaussian, private_gaussian, sample)
