#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy  # For making a full copy of objects (used when averaging model weights)
import torch  # Main PyTorch library for model building and training
from torchvision import datasets, transforms  # Load datasets like CIFAR-10 or EuroSAT, and apply transformations
from sampling import cifar_iid, eurosat_noniid  # Custom functions for splitting data between users
from torch.utils.data import random_split  # Used to divide a dataset into training and testing parts

# ------------------------------ get_dataset ------------------------------ #
def get_dataset(args):
    """
    Loads the chosen dataset (CIFAR-10 or EuroSAT), applies transformations,
    and splits it between users for training.

    Args:
        args: Settings like dataset name, number of users, and IID/non-IID option.

    Returns:
        train_dataset: Training images
        test_dataset: Test images
        user_groups: A dictionary showing which data each user gets
    """
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

        if args.iid:
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            if args.unequal:
                raise NotImplementedError("Unequal splits not supported.")
            else:
                raise NotImplementedError("Non-IID for CIFAR not supported here.")

    elif args.dataset == 'eurosat':
        data_dir = '../data/eurosat/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_dataset = datasets.EuroSAT(data_dir, download=True, transform=apply_transform)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        if args.iid:
            raise NotImplementedError("IID not supported for EuroSAT.")
        else:
            if args.unequal:
                raise NotImplementedError("Unequal splits not supported.")
            else:
                user_groups = eurosat_noniid(train_dataset, args.num_users)

    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported.")

    return train_dataset, test_dataset, user_groups

# ------------------------------ average_weights ------------------------------ #
def average_weights(w):
    """
    Averages the weights from all users' models to create a new global model.

    Args:
        w: List of model weights from each user

    Returns:
        w_avg: The new averaged model weights
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

# ------------------------------ exp_details ------------------------------ #
def exp_details(args):
    """
    Prints the main settings for this training experiment.

    Args:
        args: All the chosen settings like model name, optimizer, learning rate, etc.
    """
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning Rate : {args.lr}')
    print(f'    Global Rounds : {args.epochs}\n')

    print('    Federated Learning setup:')
    print('    Data Distribution:', 'IID' if args.iid else 'non-IID')
    print(f'    Local Batch Size : {args.local_bs}')
    print(f'    Local Epochs     : {args.local_ep}\n')
    return
