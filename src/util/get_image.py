import numpy as np
import matplotlib.pyplot as plt
import pickle
from constants import *


def unpickle(file: str) -> dict:
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def get_image(index: int) -> tuple[np.ndarray, int]:
    """get single image and label from file"""
    batch = int(index / BATCH_SIZE)
    print(batch)
    local_index = index - batch * BATCH_SIZE

    chosen_batch: dict = unpickle(f"{dataset_path}/data_batch_{batch+1}")

    return chosen_batch[b"data"][local_index], chosen_batch[b"labels"][local_index]


def show_image(index: int):
    data, _ = get_image(index)
    data = np.reshape(data, (32, 32, 3), order="F").swapaxes(0, 1)
    plt.imshow(data)
    plt.show()
