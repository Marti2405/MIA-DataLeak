import numpy as np
from tqdm import tqdm
import torch

from .utils import *
from .data_loader import DataLoader
from .resnet_architecture import ResNet
from .lenet_architecture import Net

MODEL_PATH = "../../models/"
MODEL_NAME = "baseline_resnet.pth"


class Model:
    def __init__(self, path=MODEL_PATH, name=MODEL_NAME):
        self.device = get_device()

        # load the trained model
        self.model = Net().to(self.device)
        self.model.load_state_dict(torch.load(path + name, map_location=self.device))

        # put the network in eval mode
        self.model.eval()

    def predict(self, images):
        """
        Predicts for each input image a list of probabilities associated
        with the available classes. It returns a list of lists.
        """

        # stores predicted probabilities
        predicted_prob = []

        # for each input image
        for image in tqdm(images):
            # predict logit
            y_pred = self.model(torch.from_numpy(image)[None, :, :, :].to(self.device))

            # convert tensor to numpy array
            y_pred_np = y_pred[0].cpu().detach().numpy()

            # convert logits to probabilities
            y_pred_prob = (np.exp(y_pred_np) / np.exp(y_pred_np).sum()).tolist()

            # store probabilities
            predicted_prob.append(y_pred_prob)

        return predicted_prob


if __name__ == "__main__":
    """
    This is an example that shows how to use the model.
    """

    # load data
    data_loader = DataLoader()
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()

    # instantiate the model
    model = Model()
    predicted_prob = model.predict(x_test)

    # display three predictions
    print("The predicted probabilities for the first three images:")

    for predicted_prob_ in predicted_prob[:3]:
        print(predicted_prob_)
