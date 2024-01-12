import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

DATA_PATH = "../../data/raw_dataset/"


class DataLoader:
    """
    This class downloads the CIFAR-10 dataset, preprocesses
    it and creates the train and test batches needed for training.
    """

    def __init__(self, path=DATA_PATH):
        self.path = path

        # train set pre-processor
        self.transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        # test set pre-processor
        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def load_train_data(self, batch_size=128, only_store=False):
        """
        This method loads the train data, preprocesses it and creates the data batches.
        """

        # load train data
        trainset = torchvision.datasets.CIFAR10(
            root=self.path, train=True, download=True, transform=self.transform_train
        )

        if only_store:
            return

        # define the train loader object
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)

        return trainset, trainloader

    def load_test_data(self, batch_size=128, only_store=False):
        """
        This method loads the test data, preprocesses it and creates the data batches.
        """

        # load test data
        testset = torchvision.datasets.CIFAR10(
            root=self.path, train=False, download=True, transform=self.transform_test
        )

        if only_store:
            return

        # define the test loader object
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

        return testset, testloader

    def __load_data(self, loader):
        """
        This helper method converts the loader data into a numpy array
        that has exactly the same shape as the Keras dataset.
        """

        x = []
        y = []

        for x_batch, y_batch in loader:
            data_points_no = len(x_batch)

            for index in range(data_points_no):
                x.append(x_batch[index].numpy())
                y.append([y_batch[index].numpy().item()])

        return np.array(x), np.array(y)

    def load_data(self):
        """
        Loads the CIFAR-10 dataset formatted exactly as the Keras version.
        """

        x_train, y_train = self.__load_data(self.load_train_data()[1])
        x_test, y_test = self.__load_data(self.load_test_data()[1])

        return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    """
    Store the CIFAR-10 dataset in a local folder.
    """

    print("Storing CIFAR-10 dataset locally...")

    data_loader = DataLoader()
    data_loader.load_train_data(only_store=True)
    data_loader.load_test_data(only_store=True)
