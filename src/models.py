#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
from torchvision import models


class ResNet18Model(nn.Module):
    def __init__(self, dim_out):
        super(ResNet18Model, self).__init__()
        # Load ResNet18 model from torchvision, without pretrained weights.
        self.resnet = models.resnet18(pretrained=False)
         # Get number of features from the original fully connected (fc) layer
        num_ftrs = self.resnet.fc.in_features
        # Replace the original fc layer with a dropout + linear layer for classification
        # Dropout helps prevent overfitting by randomly zeroing some inputs
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, dim_out)   # Output layer with dim_out classes
        )

    def forward(self, x):
         # Forward pass through the modified ResNet18
        x = self.resnet(x)
        # Apply log_softmax to get log-probabilities for classification
        return F.log_softmax(x, dim=1)

class CNNModel(nn.Module):
    def __init__(self, dim_out):
        super(CNNModel, self).__init__()
        # First convolutional layer:
        # Input channels: 3 (RGB image)
        # Output channels: 6 filters
        # Kernel size: 5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling layer with 2x2 window and stride 2 (reduces spatial size by half)
        self.pool = nn.MaxPool2d(2, 2)

        # Second convolutional layer:
        # Input channels: 6
        # Output channels: 16 filters
        # Kernel size: 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Fully connected layers:
        # Input features: 16 * 5 * 5 (after two conv + pool layers on 32x32 input)
        # Output features: 120 neurons
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, dim_out)

    def forward(self, x):
         # Pass input through first conv layer + ReLU activation + pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Pass through second conv layer + ReLU + pooling
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the tensor into a vector to feed fully connected layers
        x = x.view(-1, 16 * 5 * 5)

        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

         # Final output layer (logits)
        x = self.fc3(x)

        # Apply log_softmax to produce log-probabilities
        return F.log_softmax(x, dim=1)


class EuroSATCNN(nn.Module):
    def __init__(self, dim_out):
        super(EuroSATCNN, self).__init__()
        # First conv layer:
        # Input channels: 3 (RGB)
        # Output channels: 6
        # Kernel size: 5x5
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        # Max pooling with 2x2 window and stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second conv layer:
        # Input channels: 6
        # Output channels: 16
        # Kernel size: 5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # Fully connected layers:
        # Input size is adjusted for input images of size 64x64 after conv + pooling
        self.fc1 = nn.Linear(16 * 13 * 13, 120)  # Adjusted for 64x64 input
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, dim_out)

    def forward(self, x):
         # Forward through first conv + ReLU + pool
        x = self.pool(F.relu(self.conv1(x)))  # Output: 30x30x6
        x = self.pool(F.relu(self.conv2(x)))  # Output: 13x13x16
        x = x.view(-1, 16 * 13 * 13)         # Flatten: 2704 features

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer (logits)
        x = self.fc3(x)
        # Log-softmax for classification probabilities
        return F.log_softmax(x, dim=1)
