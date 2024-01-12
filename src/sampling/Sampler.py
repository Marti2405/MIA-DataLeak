import numpy as np
from constants import *


class Sampler:
    generator: np.random.Generator

    def __init__(self, seed=SEED):
        self.generator = np.random.default_rng(seed)

    def sample(
        self, percentage: float, eval_percentage=0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a fraction of the training set.
        Returns equal amount of 'private' images from the test set to calculate a model's losses.
        Images are identified by their index in the dataset. Valid indexes range from 0 to 49999.
        """
        if percentage > 20.0 or percentage < 0:
            raise Exception("invalid percentage. must be between 0 and 20.")

        sample_size = int((TRAIN_SIZE * percentage) / 100)

        train_idx = self.generator.choice(TRAIN_SIZE - 1, sample_size, replace=False)
        test_idx = self.generator.choice(TEST_SIZE - 1, sample_size, replace=False)

        size_eval = int(sample_size * eval_percentage)

        eval_train_idx = train_idx[:-size_eval]
        eval_test_idx = test_idx[:-size_eval]

        return train_idx, test_idx, eval_train_idx, eval_test_idx
