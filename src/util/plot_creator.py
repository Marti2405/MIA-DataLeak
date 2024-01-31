import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from constants import RESULTS_PATH, EXPERIMENT_NAME, LOSS_TYPE


def plot_confusion_matrix(cf, name: str):
    # Create a 2D array from the cf tuple
    cf_array = np.array([[cf[0], cf[3]], [cf[1], cf[2]]])

    # Create a seaborn heatmap
    sns.heatmap(
        cf_array,
        annot=True,
        fmt="g",
        xticklabels=["True", "False"],
        yticklabels=["True", "False"],
        cmap="crest"
    )

    # Set axis labels and title
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")

    # Save the plot
    plt.savefig(f"{RESULTS_PATH}{EXPERIMENT_NAME}/confusion_matrix_{name}.jpg", dpi=300)
    plt.show()

def plot_densities(
    train_known_loss: np.ndarray,
    train_private_loss: np.ndarray,
    x: float = None,
):
    sns.kdeplot(train_known_loss, fill=True, common_norm=False, alpha=.5, color=(0.48942421, 0.72854938, 0.56751036))
    sns.kdeplot(train_private_loss, fill=True, common_norm=False, alpha=.5, color=(0.14573579, 0.29354139, 0.49847009))
    plt.xlabel(f"{LOSS_TYPE.capitalize()}")
    plt.legend(["Known train data", "Private train data"])
    plt.title("Density of known and private train data")

    if x:
        plt.axvline(x=x, color="red", linestyle="--", alpha=0.5)
    plt.savefig(f"{RESULTS_PATH}{EXPERIMENT_NAME}/density.jpg", dpi=300)
    plt.show()

def plot_loss_arrays(arr1, arr2):
    """
    Plot both loss arrays - histograms
    Input: Loss arrays for Known/ Unknown Datasets
    Output: Plot of both losses
    """
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    fig.set_size_inches(10, 6)

    axs[0].hist(arr1, bins=20, color=(0.48942421, 0.72854938, 0.56751036))
    axs[0].set_title("Known train data")
    axs[0].set_xlabel(f"{LOSS_TYPE}")
    axs[0].set_ylabel("Count")

    axs[1].hist(arr2, bins=20, color=(0.14573579, 0.29354139, 0.49847009))
    axs[1].set_title("Private train data")
    axs[1].set_xlabel(f"{LOSS_TYPE}")

    fig.suptitle(f"Histograms of the {LOSS_TYPE} losses")

    plt.savefig(f"{RESULTS_PATH}{EXPERIMENT_NAME}/losses.jpg", dpi=300)
    plt.show()
