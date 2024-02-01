from training.model import Model
import numpy as np
from constants import EPSILON

class LossCalculator:
    model: Model

    def __init__(self, model, loss_type):
        self.model = model
        # function that computes the type of chosen loss
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

        loss_known = self.compute_loss(predictions_prob_known)
        loss_unknown = self.compute_loss(predictions_prob_unknown)

        return loss_known, loss_unknown
    
    def compute_loss(self, predictions_prob):        
        if self.loss_type == "probability":
            return np.array([prob for prob in predictions_prob])
        elif self.loss_type == "cross_entropy":
            return np.array([-np.log(prob) for prob in predictions_prob])
        elif self.loss_type == "normalized_probability":
            return np.log(predictions_prob / (1 - predictions_prob + EPSILON))
        else:
            raise Exception("This type of loss it not defined.")