import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_pil_image, to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_images(image_paths, image_labels, num_images):
    """
    Plotting a sample images of the dataset randomly

    Args:
    - image_paths (list): List of paths to the images.
    - image_labels (list): List of labels corresponding to each image.
    - num_images (int): Number of random images to display.
    """
    # Create a list of tuples (image_path, label)
    images_with_labels = list(zip(image_paths, image_labels))

    # Randomly select the specified number of images
    selected_images = random.sample(
        images_with_labels, min(num_images, len(images_with_labels))
    )

    plt.figure(figsize=(15, 10))
    for i, (image_path, label) in enumerate(selected_images):
        image = Image.open(image_path)
        plt.subplot(4, 5, i + 1)  # Arrange in a grid of 4 rows x 5 columns
        plt.imshow(image)
        plt.title(f"Label: {label}", fontsize=10)
        plt.axis("off")  # Turn off axes for better visualization
    plt.tight_layout()
    plt.show()


def visualize_samples(dataset, num_samples):
    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        image, label = dataset[i]
        image = image.permute(
            1, 2, 0
        ).numpy()  # Convert Tensor (C, H, W) to NumPy (H, W, C)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(
            np.clip(image, 0, 1)
        )  # Clip values to range [0, 1] for visualization
        plt.title(f"Label: {label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_generated_images(generator, num_images, z_dim):
    generator.eval()  # Set the generator to evaluation mode
    with torch.no_grad():
        z = torch.randn(num_images, z_dim, device=device)  # Random noise
        generated_images = generator(z).detach().cpu()
        generated_images = (
            generated_images + 1
        ) / 2  # Denormalize from [-1, 1] to [0, 1]

    # Plot the generated images
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        img = to_pil_image(generated_images[i])  # Convert tensor to PIL image
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
