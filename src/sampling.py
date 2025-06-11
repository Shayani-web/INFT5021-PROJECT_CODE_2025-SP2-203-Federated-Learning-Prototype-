#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6+

# === Core Libraries ===
import numpy as np  # Used for array manipulations, random selection, indexing

# === TorchVision Datasets and Transforms ===
from torchvision import datasets, transforms  # Used to load datasets like CIFAR-10 and EuroSAT


def cifar_iid(dataset, num_users):
    """
    Splits the CIFAR-10 dataset into IID (Independent and Identically Distributed) partitions.

    Each user will receive approximately an equal number of randomly selected images from all classes.
    This simulates an ideal data distribution where each user has the same kind of data.

    Args:
        dataset: CIFAR-10 dataset (already loaded with torchvision)
        num_users (int): Number of simulated clients/users

    Returns:
        dict_users (dict): Dictionary where key = user ID, value = list of image indices assigned to that user

    Variables:
        num_items: Total items per user = total dataset size / num_users
        all_idxs: List of all available image indices
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))  # Sample without replacement
        all_idxs = list(set(all_idxs) - dict_users[i])  # Remove assigned indices from pool

    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Splits the CIFAR-10 dataset into non-IID (non-Independent and Identically Distributed) partitions.

    Each client receives data from only a few classes (e.g., 2), creating a skewed label distribution.
    This simulates real-world scenarios where users might not have data from every class.

    Args:
        dataset: CIFAR-10 dataset with class labels
        num_users (int): Number of simulated clients/users

    Returns:
        dict_users (dict): Mapping of user IDs to selected image indices (non-IID)

    Notes:
        - CIFAR-10 has 10 classes
        - Dataset is split into 200 shards of 250 images each
        - Each user is assigned 2 random shards → gets 500 images total
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]  # Shard IDs: 0 to 199
    dict_users = {i: np.array([]) for i in range(num_users)}  # Initialize user data dict
    idxs = np.arange(num_shards * num_imgs)  # Full index list (0 to 49999)
    labels = np.array(dataset.train_labels)  # CIFAR-10 labels

    # Sort indices by labels to group similar-class images together
    idxs_labels = np.vstack((idxs, labels))  # Stack indices and labels
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # Sort by label
    idxs = idxs_labels[0, :]  # Get sorted indices only

    # Assign 2 random shards to each user (non-IID)
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))  # Pick 2 random shards
        idx_shard = list(set(idx_shard) - rand_set)  # Remove selected shards from pool

        for rand in rand_set:
            start = rand * num_imgs
            end = (rand + 1) * num_imgs
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:end]), axis=0)

    return dict_users


def eurosat_noniid(dataset, num_users):
    """
    Splits the EuroSAT (RGB) dataset into non-IID partitions.

    Each user receives images only from 2 randomly chosen classes. This simulates a scenario
    where each user (e.g., satellite or regional sensor) sees only specific land cover types.

    Args:
        dataset: EuroSAT dataset (already loaded, possibly using Subset wrapper from DataLoader)
        num_users (int): Number of simulated users/clients

    Returns:
        dict_users (dict): Dictionary mapping user ID → list of image indices (non-IID)

    Key Variables:
        num_classes: Number of classes in EuroSAT (10)
        num_shards_per_class: Number of partitions per class (e.g., 20)
        num_imgs_per_shard: Number of images in each shard
    """
    num_classes = 10
    num_shards_per_class = 20  # 20 * 10 = 200 total shards
    num_imgs_per_shard = len(dataset) // (num_classes * num_shards_per_class)

    idx_shard = [i for i in range(num_classes * num_shards_per_class)]  # Shard IDs: 0–199
    dict_users = {i: np.array([]) for i in range(num_users)}  # Init user dictionary
    idxs = np.arange(len(dataset))  # All indices

    # Extract labels — assumes dataset is a torch.utils.data.Subset
    labels = np.array(dataset.dataset.targets)[dataset.indices]

    # Sort indices by label for grouping by class
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]  # Sorted indices

    # Assign 2 classes to each user and select one shard per class
    for i in range(num_users):
        user_classes = np.random.choice(num_classes, size=2, replace=False)  # Choose 2 distinct classes

        for cls in user_classes:
            shard_range = list(range(cls * num_shards_per_class, (cls + 1) * num_shards_per_class))
            rand_shard = np.random.choice(shard_range, size=1, replace=False)[0]

            start_idx = rand_shard * num_imgs_per_shard
            end_idx = start_idx + num_imgs_per_shard

            dict_users[i] = np.concatenate((dict_users[i], idxs[start_idx:end_idx]), axis=0)

    return dict_users
