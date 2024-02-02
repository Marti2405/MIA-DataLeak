import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from constants import RESULTS_PATH, EXPERIMENT_NAME, LOSS_TYPE, STORE_INTERMEDIATE_FIGURES, PLOT_INTERMEDIATE_FIGURES


def plot_confusion_matrix(cf, name: str, percentage: int):
    # Create a 2D array from the cf tuple
    cf_array = np.array([[cf[0], cf[3]], [cf[1], cf[2]]])

    # Create a seaborn heatmap
    sns.heatmap(
        cf_array/np.sum(cf_array),
        annot=True,
        fmt='.2%',
        xticklabels=["True", "False"],
        yticklabels=["True", "False"],
        cmap="crest"
    )

    # Set axis labels and title
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Averaged {name} confusion matrix\n{percentage}% of leaked training data")

    # Save the plot
    plt.savefig(f"{RESULTS_PATH}{EXPERIMENT_NAME}/confusion_matrix_avg_{name}_{percentage}%.jpg", dpi=300)
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def plot_densities(
    train_known_loss: np.ndarray,
    train_private_loss: np.ndarray,
    percentage: int,
    dataset: str,
    x: float = None,
):
    sns.kdeplot(train_known_loss, fill=True, common_norm=False, alpha=.5, color=(0.48942421, 0.72854938, 0.56751036))
    sns.kdeplot(train_private_loss, fill=True, common_norm=False, alpha=.5, color=(0.14573579, 0.29354139, 0.49847009))
    plt.xlabel(f"{LOSS_TYPE.capitalize()}")
    plt.legend(["Known training data", "Private data"])
    plt.title(f"Density of known training data and private data\nused for {dataset} - {percentage}% of leaked training data")

    if x:
        plt.axvline(x=x, color="red", linestyle="--", alpha=0.5)
    if STORE_INTERMEDIATE_FIGURES:
        plt.savefig(f"{RESULTS_PATH}{EXPERIMENT_NAME}/{dataset}_densities_sample_{percentage}%.jpg", dpi=300)
    if PLOT_INTERMEDIATE_FIGURES:
        plt.show(block=False)
        plt.pause(3)
    plt.close()

def plot_loss_arrays(arr1, arr2, percentage):
    """
    Plot both loss arrays - histograms
    Input: Loss arrays for Known/ Unknown Datasets
    Output: Plot of both losses
    """
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    fig.set_size_inches(10, 6)

    axs[0].hist(arr1, bins=20, color=(0.48942421, 0.72854938, 0.56751036))
    axs[0].set_title("Known training data")
    axs[0].set_xlabel(f"{LOSS_TYPE.capitalize()}")
    axs[0].set_ylabel("Count")

    axs[1].hist(arr2, bins=20, color=(0.14573579, 0.29354139, 0.49847009))
    axs[1].set_title("Private data")
    axs[1].set_xlabel(f"{LOSS_TYPE}")

    fig.suptitle(f"Histograms of the {LOSS_TYPE} losses\n{percentage}% of leaked training data")

    if STORE_INTERMEDIATE_FIGURES:
        plt.savefig(f"{RESULTS_PATH}{EXPERIMENT_NAME}/losses_sample_{percentage}%.jpg", dpi=300)
    if PLOT_INTERMEDIATE_FIGURES:
        plt.show(block=False)
        plt.pause(3)
    plt.close()


def plot_wasserstein_dist(percentages, wasserstein_dist_values):
    """
    Plot and store a scatter plot of the Wasserstein distance values.
    """

    # create the scatter plot
    plt.plot(percentages, wasserstein_dist_values, marker="o", color=(0.48942421, 0.72854938, 0.56751036))
    plt.title(f"Average Wasserstein distance between the {LOSS_TYPE}\ndistributions of leaked training and private data",)
    plt.xlabel("Percentage of known training data")
    plt.ylabel("Average Wasserstein distance")

    # save the plot as a PNG image
    if STORE_INTERMEDIATE_FIGURES:
        plt.savefig(f"{RESULTS_PATH}{EXPERIMENT_NAME}/wasserstein_distance.jpg", dpi=300)

    plt.show(block=False)
    plt.pause(3)
    plt.close()
