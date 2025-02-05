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

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Creating Gaussian Noise
class GaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=0.1):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise


# Adding a layer of Distortion to the images using this class
class Distortion(nn.Module):
    def __init__(self):
        super(Distortion, self).__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            ]
        )

    def forward(self, x):
        x_distorted = torch.stack(
            [to_tensor(self.transform(to_pil_image(img))) for img in x]
        ).to(x.device)
        return x_distorted


# Defining GAN Model


# Generator Network
class Generator(nn.Module):

    def __init__(self, z_dim=100):

        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * 64 * 3),
            nn.Tanh(),  # Output values in [-1, 1]
        )

        self.noise_layer = GaussianNoise(mean=0.0, std=0.1)

        self.distortion_layer = Distortion()

    def forward(self, z):

        img = self.model(z)

        img = img.view(-1, 3, 64, 64)

        img = self.noise_layer(img)  # Add noise

        img = self.distortion_layer(img)  # Add distortions

        return img


# Discriminator Network
class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):

        img_flat = img.view(img.size(0), -1)

        validity = self.model(img_flat)

        return validity


def fgsm_attack(images, labels, model, epsilon=0.03):
    images.requires_grad = True
    outputs = model(images)  # Forward pass through the model

    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()  # Compute gradients with respect to the inputs

    # Generate adversarial examples by applying FGSM
    perturbed_images = images + epsilon * images.grad.sign()
    perturbed_images = torch.clamp(
        perturbed_images, -1, 1
    )  # Clip pixel values to valid range
    return perturbed_images


# preprocessing function for additive noise and color jitter
def custom_preprocessing(images):
    """
    Apply additive noise and color jitter transformations to the input images.
    """
    transformed_images = []
    for img in images:
        pil_img = to_pil_image(img)  # Convert tensor to PIL image
        transform = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
                transforms.ToTensor(),
            ]
        )
        transformed_img = transform(pil_img)  # Apply the transformations

        # Add light additive noise
        noise = torch.randn_like(transformed_img) * 0.02
        transformed_img = transformed_img + noise
        transformed_img = torch.clamp(transformed_img, 0, 1)

        transformed_images.append(transformed_img)
    return torch.stack(transformed_images).to(images.device)


def pgd_attack_with_preprocessing(
    images,
    labels,
    model,
    preprocess_fn,
    epsilon,
    alpha,
    iterations,
):
    """
    Perform PGD attack with preprocessing applied to the input images.
    """
    # Apply preprocessing transformations
    preprocessed_images = preprocess_fn(images)

    # Clone images for adversarial attack
    perturbed_images = (
        preprocessed_images.clone()
        .detach()
        .requires_grad_(True)
        .to(preprocessed_images.device)
    )

    for _ in range(iterations):
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        # Normalize gradients and add perturbations
        grad = perturbed_images.grad.data
        grad = grad / (grad.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-10)

        perturbed_images = perturbed_images + alpha * grad.sign()

        # Project perturbations to epsilon-ball
        perturbations = torch.clamp(
            perturbed_images - preprocessed_images, -epsilon, epsilon
        )
        perturbed_images = (
            torch.clamp(preprocessed_images + perturbations, 0, 1)
            .detach()
            .requires_grad_(True)
        )

    return perturbed_images
