#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
from torchvision import models


class ResNet18Model(nn.Module):
    def __init__(self, dim_out):
        super(ResNet18Model, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, dim_out)
        )

    def forward(self, x):
        x = self.resnet(x)
        return F.log_softmax(x, dim=1)

class CNNModel(nn.Module):
    def __init__(self, dim_out):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, dim_out)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class EuroSATCNN(nn.Module):
    def __init__(self, dim_out):
        super(EuroSATCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)  # Adjusted for 64x64 input
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, dim_out)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Output: 30x30x6
        x = self.pool(F.relu(self.conv2(x)))  # Output: 13x13x16
        x = x.view(-1, 16 * 13 * 13)         # Flatten: 2704 features
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
