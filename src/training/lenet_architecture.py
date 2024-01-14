import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    This class defines a modified architecture of the LeNet-5 model. 
    """

    def __init__(self, in_channels=3, out_size=10):
        """
        Defines the layers used by the network.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_size)

    def forward(self, x):
        """
        This method connects the layers.
        """

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
