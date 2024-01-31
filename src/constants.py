SEED: int = 131574130126691614769715556035217985886
dataset_path: str = "../../data/cifar-10-batches-py"
TRAIN_SIZE: int = 50000
TEST_SIZE: int = 10000
BATCH_SIZE: int = 10000
EPSILON = 1E-31
# define the type of loss
LOSS_TYPE = "probability"
# define results folder path
RESULTS_PATH = "../results/"
DATASET_PATH = "../data/raw_dataset/"
EXPERIMENT_NAME = f'lenet_100_epochs_{LOSS_TYPE}_loss'