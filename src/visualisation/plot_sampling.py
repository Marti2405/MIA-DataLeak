from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import random




########### Plot PCA with Sample ########################################################
# Around 5sec to run

def plot_PCA_with_sample(all_data, sampled_data, point_size=10, alpha_value=0.5):
    # Separate the input tuples into x and y data
    all_x_data, all_y_data = all_data
    sampled_x_data, sampled_y_data = sampled_data

    # Normalize the data
    all_x_data_normalized = all_x_data / 255.0
    sampled_x_data_normalized = sampled_x_data / 255.0

    # Flatten images in the datasets
    all_flat = all_x_data_normalized.reshape(all_x_data_normalized.shape[0], -1)
    sampled_flat = sampled_x_data_normalized.reshape(sampled_x_data_normalized.shape[0], -1)

    # Perform PCA on all_data
    pca_all = PCA(n_components=2)
    principal_all = pca_all.fit_transform(all_flat)

    # Perform PCA on sampled_data
    pca_sampled = PCA(n_components=2)
    principal_sampled = pca_sampled.fit_transform(sampled_flat)

    # Create a scatterplot for the PCA space
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=principal_all[:, 0], y=principal_all[:, 1],
        hue="Label",
        palette=sns.color_palette("Set2", 10),
        data=pd.DataFrame({
            "Principal Component 1": principal_all[:, 0],
            "Principal Component 2": principal_all[:, 1],
            "Label": all_y_data.flatten()
        }),
        legend="full",
        alpha=1
    )

    # Plot the sampled data with adjusted size and alpha
    plt.scatter(
        principal_sampled[:, 0], principal_sampled[:, 1],
        color="black",
        marker="o",
        label="Sample",
        s=point_size,  # Adjust the point size
        alpha=alpha_value,  # Adjust the transparency
    )

    plt.title(f"PCA Visualization with Sample in Black")
    plt.show()

# Example usage:
#
# Assuming x_train contains the entire dataset and y_train contains the corresponding labels
# and sampled_x_train, sampled_y_train are the sampled data
#
# plot_PCA_with_sample((x_train, y_train), (sampled_x_train, sampled_y_train))

################################################################################################################
################################################################################################################



########### Plot t-SNE with Sample ########################################################
# Around 7min to run

def plot_TSNE_with_sample(all_data, sampled_data, point_size=10, alpha_value=0.5):
    # Separate the input tuples into x and y data
    all_x_data, all_y_data = all_data
    sampled_x_data, sampled_y_data = sampled_data

    # Normalize the data
    all_x_data_normalized = all_x_data / 255.0
    sampled_x_data_normalized = sampled_x_data / 255.0

    # Flatten images in the datasets
    all_flat = all_x_data_normalized.reshape(all_x_data_normalized.shape[0], -1)
    sampled_flat = sampled_x_data_normalized.reshape(sampled_x_data_normalized.shape[0], -1)

    # Perform t-SNE on all_data
    tsne_all = TSNE(n_components=2)
    principal_all = tsne_all.fit_transform(all_flat)

    # Perform t-SNE on sampled_data
    tsne_sampled = TSNE(n_components=2)
    principal_sampled = tsne_sampled.fit_transform(sampled_flat)

    # Create a scatterplot for the t-SNE space
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=principal_all[:, 0], y=principal_all[:, 1],
        hue="Label",
        palette=sns.color_palette("Set2", 10),
        data=pd.DataFrame({
            "TSNE Component 1": principal_all[:, 0],
            "TSNE Component 2": principal_all[:, 1],
            "Label": all_y_data.flatten()
        }),
        legend="full",
        alpha=0.7
    )

    # Plot the sampled data with adjusted size and alpha
    plt.scatter(
        principal_sampled[:, 0], principal_sampled[:, 1],
        color="black",
        marker="o",
        label="Sample",
        s=point_size,  # Adjust the point size
        alpha=alpha_value,  # Adjust the transparency
    )

    plt.title(f"t-SNE Visualization with Sample")
    plt.show()

# Example usage:
#
# Assuming x_train contains the entire dataset and y_train contains the corresponding labels
# and sampled_x_train, sampled_y_train are the sampled data
#
# plot_TSNE_with_sample((x_train, y_train), (sampled_x_train, sampled_y_train))
################################################################################################################
################################################################################################################
    


########### Random percentage sampler ########################################################

def random_percentage_sample(x_data, y_data, sample_percentage=0.05):
    # Check if x_data and y_data have the same length
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length.")

    num_samples = int(len(x_data) * sample_percentage)

    # Generate random indices for sampling
    random_indices = np.random.choice(len(x_data), num_samples, replace=False)

    # Extract the sampled data and labels based on the random indices
    sampled_x_data = x_data[random_indices]
    sampled_y_data = y_data[random_indices]

    return sampled_x_data, sampled_y_data

#     Example usage:
#     
#     Assuming x_train contains the data and y_train contains the labels
#
#     sampled_x_train, sampled_y_train = random_percentage_sample(x_train, y_train, sample_percentage=0.05)

################################################################################################################
################################################################################################################



if __name__=="__main__":
    #load dataset from cifar with keras library and load the dataset and store in keras directory
    pic_class = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = pic_class.load_data()

    # Draw x% of the Dataset
    sampled_x_train, sampled_y_train = random_percentage_sample(x_train, y_train, sample_percentage=0.05)

    # Plot PCA and t-SNE of the Sampled Dataset and General Dataset
    plot_PCA_with_sample((x_train, y_train), (sampled_x_train, sampled_y_train)) # 5sec to run
    plot_TSNE_with_sample((x_train, y_train), (sampled_x_train, sampled_y_train)) # 7min to run
    