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


def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return correct / total


def evaluate_adversarial(model, generator, dataloader, attack_fn, epsilon):
    model.eval()
    generator.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        perturbed_images = attack_fn(images, labels, model, epsilon)
        noise_vector = torch.randn(
            perturbed_images.size(0), 100, device=device
        )
        gan_generated_images = generator(noise_vector)
        distorted_images = (gan_generated_images + perturbed_images) / 2
        outputs = model(distorted_images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return correct / total


def evaluate_adversarial2(
    model, dataloader, attack_fn, preprocess_fn, epsilon, alpha, iterations
):

    model.eval()
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial examples using the attacking model
        perturbed_images = attack_fn(
            images, labels, model, preprocess_fn, epsilon, alpha, iterations
        )

        # Evaluate the model on adversarial examples
        outputs = model(perturbed_images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return correct / total
