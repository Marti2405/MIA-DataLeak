import numpy as np
from training.model import Model


class MembershipPredictorKDE:
    model: Model

    def __init__(self, known_density, private_density):
        self.known_density = known_density
        self.private_density = private_density

    def predict(self, loss: np.ndarray) -> np.ndarray:
        """
        This method returns ratio value. A higher value means higher likelihood
        of being part of training dataset. A value of 1 means equal.
        """

        p_from_known_density = np.array(
            [self.known_density.get_density().evaluate(x) for x in loss]
        )
        p_from_unknown_density = np.array(
            [self.private_density.get_density().evaluate(x) for x in loss]
        )

        result = p_from_known_density / (p_from_known_density + p_from_unknown_density)

        return (result > 0.5).astype(int)
