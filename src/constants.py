EPSILON = 1E-31
TRAIN_SIZE: int = 50000
TEST_SIZE: int = 10000
BATCH_SIZE: int = 10000
SEED: int = 131574130126691614769715556035217985886

# define parameters for experiments
LOSS_TYPE = ["probability", "cross entropy", "normalized probability"][0]
PERCENTAGES = list(range(1, 17))
MODEL_NAME = "/baseline_lenet5_100_epochs.pth"
TRIALS = 10

# define data paths
RESULTS_PATH = "../results/"
DATASET_PATH = "../data/raw_dataset/"
EXPERIMENT_NAME = f'lenet_100_epochs_{LOSS_TYPE}_loss'
INPUT_DATASET_PATH: str = "../../data/cifar-10-batches-py"

# set options about plotting and figure storage
PLOT_INTERMEDIATE_FIGURES = False
STORE_INTERMEDIATE_FIGURES = True
