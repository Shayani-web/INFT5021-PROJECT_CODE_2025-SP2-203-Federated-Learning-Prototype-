import os
import time
import math
import copy
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from options import args_parser
from update import LocalUpdate, test_inference
from utils import get_dataset, average_weights, exp_details
from torch.utils.data import DataLoader, random_split, Subset
import random
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def create_clients(dataset, num_clients=40, num_orbits=5):
    indices = list(range(len(dataset)))
    random.shuffle(indices)  # Ensure data is randomly distributed
    split_indices = np.array_split(indices, num_clients)

    clients = [Subset(dataset, idxs) for idxs in split_indices]

    # Block-wise orbit assignment: group nearby clients into the same orbit
    clients_per_orbit = num_clients // num_orbits
    orbits = [i for i in range(num_orbits) for _ in range(clients_per_orbit)]

    return clients, orbits




def async_aggregate(global_model, local_weights, alpha=0.5):
    """
    I’m blending the local model from one client into the global model.
    The 'alpha' value controls how much of the local model I want to mix in.
    If alpha = 0.5, it means equally weighting the local and global models.
    """
    # First, grab the current weights from the global model
    global_weights = global_model.state_dict()

    # For every layer in the model, combine the old global weight with the new client weight
    for key in global_weights:
        global_weights[key] = alpha * local_weights[key] + (1 - alpha) * global_weights[key]

    # Then load these newly combined weights back into the global model
    global_model.load_state_dict(global_weights)

    # Finally, return the updated global model so it can be used for the next round or by other clients
    return global_model


def fedAsync_Training(args, train_dataset, test_dataset, user_groups, global_model, logger, device):
    """Implement hierarchical FL with visibility-based timing and energy computation."""
    device = 'cuda' if args.gpu else 'cpu'
    global_model.to(device)
    global_model.train()
    
    # Initialize metrics
    train_accuracy = []

    clients = create_clients(train_dataset, num_clients=40)

    # Loop through the number of training rounds
    for rnd in range(num_rounds):
        print(f"\n--- Round {rnd + 1} ---")  # track progress

        # Each round, simulates asynchronous behavior by picking one random client (instead of all clients training in sync)
        selected_client = random.choice(range(len(clients)))
        print(f"Selected Client: {selected_client}")  # Show which client was picked

        # Create a fresh copy of the global model for the client to train on
        local_model = copy.deepcopy(global_model)  # Make a copy of the global model so the client can train on it
        local_model.load_state_dict(global_model.state_dict())  # Start the client with the latest global model weights

        # Now train this local model on the client's data
        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[selected_client], logger=logger)
                
        updated_weights, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=rnd)
        

        # After training, send the updated weights to the central server
        # The global model is updated using this client's new weights (alpha controls how much influence the update has)
        global_model = async_aggregate(global_model, updated_weights, alpha=0.5)

        list_acc = []
        global_model.eval()
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[selected_client], logger=logger)
            acc, _ = local_model.inference(model=global_model)
            list_acc.append(acc)
        train_accuracy.append(sum(list_acc) / len(list_acc))

    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    
    # Print final results
    print(f'\n Results after {args.epochs} global rounds:')
    print(f"|---- Avg Train Accuracy: {100 * train_accuracy[-1]:.2f}%")
    print(f"|---- Test Accuracy: {100 * test_acc:.2f}%")

    return train_accuracy, test_acc, test_loss
