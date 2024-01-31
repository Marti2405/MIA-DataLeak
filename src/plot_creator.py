import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_densities(
    train_known_loss: np.ndarray,
    train_private_loss: np.ndarray,
    results_path: str,
    experiment_name: str,
    x: float = None,
):
    sns.kdeplot(train_known_loss)
    sns.kdeplot(train_private_loss)

    if x:
        plt.axvline(x=x, color="red", linestyle="--", alpha=0.5)
    plt.savefig(f"{results_path}{experiment_name}/density.jpg", dpi=300)
    plt.show()
