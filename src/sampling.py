#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms



def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.train_labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def eurosat_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from EuroSAT dataset
    :param dataset: EuroSAT dataset (e.g., loaded via torchvision.datasets.EuroSAT)
    :param num_users: Number of users
    :return: dict of image indices for each user
    """
    # EuroSAT has 10 classes; adjust shards per class
    num_classes = 10
    num_shards_per_class = 20  # 20 shards per class, total 200 shards
    num_imgs_per_shard = len(dataset) // (num_classes * num_shards_per_class)  # e.g., 27,000 images / 200 = 135
    
    idx_shard = [i for i in range(num_classes * num_shards_per_class)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.dataset.targets)[dataset.indices] # Assuming dataset.targets provides labels
    
    # Sort indices by labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    # Assign shards to users based on a subset of classes
    for i in range(num_users):
        # Each user gets data from 2 random classes (non-IID)
        user_classes = np.random.choice(num_classes, size=2, replace=False)
        for cls in user_classes:
            # Select one shard randomly from this class
            shard_indices = list(range(cls * num_shards_per_class, (cls + 1) * num_shards_per_class))
            rand_shard = np.random.choice(shard_indices, size=1, replace=False)[0]
            start_idx = rand_shard * num_imgs_per_shard
            end_idx = start_idx + num_imgs_per_shard
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[start_idx:end_idx]), axis=0)
    
    return dict_users


