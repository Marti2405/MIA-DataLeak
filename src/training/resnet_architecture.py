import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    This class defines the architecture of a residual block
    and the function needed to pass the information through it.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Defines the layers used in the block.
        """

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.batch1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.batch2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip_connection = lambda x: x

    def forward(self, x):
        """
        This method connects the layers into one block.
        """

        skip = self.skip_connection(x)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x + skip)

        return x


class ResNet(nn.Module):
    """
    It defines the architecture of the ResNet-18 network.
    """

    def __init__(self, in_channels=3, out_size=10):
        """
        This method creates all the layers and initializes the residual blocks.
        """

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.batch1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(64, 64, 64),
                ResidualBlock(64, 64, 64),
                ResidualBlock(64, 128, 128),
                ResidualBlock(128, 128, 128),
                ResidualBlock(128, 256, 256),
                ResidualBlock(256, 256, 256),
                ResidualBlock(256, 512, 512),
                ResidualBlock(512, 512, 512),
            ]
        )

        self.dense_layer = nn.Linear(512, out_size)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )

    def forward(self, x):
        """
        Connects the layers and residual blocks.
        """

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.pool1(x)

        for block in self.res_blocks:
            x = block.forward(x)

        x = F.avg_pool2d(x, x.shape[2:])

        x = x.view(x.size(0), -1)
        x = self.dense_layer(x)

        return x
