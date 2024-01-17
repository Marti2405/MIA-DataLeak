import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    This class defines a modified architecture of the LeNet-5 model. 
    It has been implemented using:
    https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-lenet5-cifar10.ipynb.
    """

    def __init__(self, in_channels=3, out_size=10, filter_multiplier=1):
        """
        Defines the layers used by the network. The filter multiplier should be
        1 for the original LeNet5 and 3 for the improved version that accounts for
        RGB input images.
        """
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6 * filter_multiplier, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6 * filter_multiplier, 16 * filter_multiplier, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5 * filter_multiplier, 120 * filter_multiplier)
        self.fc2 = nn.Linear(120 * filter_multiplier, 84 * filter_multiplier)
        self.fc3 = nn.Linear(84 * filter_multiplier, out_size)

    def forward(self, x):
        """
        This method connects the layers.
        """

        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        
        return x
