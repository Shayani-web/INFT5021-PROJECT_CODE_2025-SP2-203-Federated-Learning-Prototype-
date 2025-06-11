#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6+

# === PyTorch Core Modules ===
from torch import nn  # Defines neural network layers like Conv2D, Linear
import torch.nn.functional as F  # Provides activation functions like relu(), log_softmax()

# === TorchVision Prebuilt Models ===
from torchvision import models  # For using ResNet18 and CNN

# ---------------------------------------------------------------------------------------
# MODEL 1: Modified ResNet18 for EuroSAT (RGB Only)
# ---------------------------------------------------------------------------------------

class ResNet18Model(nn.Module):
    """
    A modified ResNet18 model for RGB image classification.

    This model is adapted to work with 3-channel RGB images (e.g., EuroSAT RGB version).
    The final fully connected layer is replaced to output predictions for the desired number of classes.

    Args:
        dim_out (int): Number of output classes (e.g., 10 for EuroSAT).

    Returns:
    Tensor: Log-scaled predictions showing how likely the input belongs to each class.
    """

    def __init__(self, dim_out):
        super(ResNet18Model, self).__init__()

        # Load standard ResNet18 (no pretrained weights for fair training)
        self.resnet = models.resnet18(pretrained=False)

        # Modify input layer if needed (defaults to 3 channels which is fine for RGB)
        # Note: If loading 13-band images later, modify here to accept 13 input channels

        # Replace the final FC layer with dropout + linear output for 'dim_out' classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),               # Prevents overfitting
            nn.Linear(num_ftrs, dim_out)     # New final layer
        )

    def forward(self, x):
        """
    Runs the input images through the model to make predictions.

    Args:
        x (Tensor): Batch of images (e.g., shape: batch_size x 3 x height x width)

    Returns:
        Tensor: Log-scale scores showing how likely each class is.
        """
        x = self.resnet(x)
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------------------
# MODEL 2: Simple CNN (RGB image classification task for CIFAR-10)
# ---------------------------------------------------------------------------------------

class CNNModel(nn.Module):
    """
    A simple CNN for classifying small RGB images like CIFAR-10 or resized EuroSAT.

    Structure:
    - 2 convolutional layers + max pooling
    - 3 fully connected (dense) layers

    Args:
        dim_out (int): Number of output classes.

    Input:
        Tensor of shape (batch_size, 3, 32, 32)

    Output:
        Log-probabilities for each class
    """

    def __init__(self, dim_out):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)     # Input: 3 channels (RGB)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)    # Output: 16 feature maps

        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16 filters × 5×5 region = 400 features
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, dim_out)      # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # Output: (6, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # Output: (16, 5, 5)
        x = x.view(-1, 16 * 5 * 5)              # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------------------
# MODEL 3: EuroSAT CNN (Image classification for 64x64 RGB images, 10 classes)
# ---------------------------------------------------------------------------------------

class EuroSATCNN(nn.Module):
    """
    A CNN tailored for RGB version of the EuroSAT dataset (images are 64x64 and 3-channel RGB).

    EuroSATCNN Architecture:
    - 2 convolutional layers
    - Max pooling for spatial reduction
    - 3 fully connected layers

    Args:
        dim_out (int): Number of output land cover classes (EuroSAT RGB has 10).

    Input:
        Tensor of shape (batch_size, 3, 64, 64)

    Output:
        Log-probabilities for 10 classes
    """

    def __init__(self, dim_out):
        super(EuroSATCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)      # Input: RGB → 6 filters
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)     # Output: (16, 26, 26) → pooled: (16, 13, 13)

        self.fc1 = nn.Linear(16 * 13 * 13, 120)  # Flattened features = 2704
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, dim_out)        # Output layer: 10 land cover classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # Output: (6, 30, 30)
        x = self.pool(F.relu(self.conv2(x)))     # Output: (16, 13, 13)
        x = x.view(-1, 16 * 13 * 13)             # Flatten to vector
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
