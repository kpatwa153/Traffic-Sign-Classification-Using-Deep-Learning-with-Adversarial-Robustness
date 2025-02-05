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


def pgd_attack(model, images, labels, epsilon, alpha, attack_epochs):
    # Clone images and enable gradient computation
    perturbed_images = images.clone().detach()
    perturbed_images.requires_grad = True

    for _ in range(attack_epochs):
        # Forward pass
        outputs = model(perturbed_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # Zero gradients
        model.zero_grad()

        # Backward pass to compute gradients
        loss.backward()

        # Ensure gradient is not None
        if perturbed_images.grad is None:
            raise RuntimeError(
                "Gradient computation failed. Check requires_grad settings."
            )

        # Update perturbed images
        perturbed_images = (
            perturbed_images + alpha * perturbed_images.grad.sign()
        )

        # Clip perturbed images to be within epsilon-ball of original images
        perturbed_images = torch.clamp(
            perturbed_images, images - epsilon, images + epsilon
        )

        # Clip values to valid image range [0, 1]
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        # Detach and re-enable gradient computation for the next iteration
        perturbed_images = perturbed_images.clone().detach()
        perturbed_images.requires_grad = True

    return perturbed_images
