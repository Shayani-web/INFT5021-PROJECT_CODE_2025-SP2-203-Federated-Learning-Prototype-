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
    Blending the local model from one client into the global model.
    The 'alpha' value controls how much of the local model I want to mix in.
    If alpha = 0.5, it means equally weighting the local and global models.
    """
    # First, grab the current weights from the global model
    global_weights = global_model.state_dict()

    # For every layer in the model, combine the old global weight with the new client weight
    for key in global_weights:
        global_weights[key] = alpha * local_weights[key] + (1 - alpha) * global_weights[key]

    # Then load these newly combined weights back into the global model.
    global_model.load_state_dict(global_weights)

    # Finally, return the updated global model so it can be used for the next round or by other clients
    return global_model

def estimate_energy_consumption(duration_sec, device='cpu'):
    """
    Estimate energy (in Joules) consumed during a training round.
    Adjust power consumption values based on actual hardware.
    """
    # Example average power draw: GPU = 80W, CPU = 20W
    power_draw_watts = 80 if device == 'cuda' else 20
    energy_joules = power_draw_watts * duration_sec  # Energy = Power × Time
    return energy_joules

def fedAsync_Training(args, train_dataset, test_dataset, user_groups, global_model, logger, device):
    device = 'cuda' if args.gpu else 'cpu'
    global_model.to(device)
    global_model.train()
    
    # Initialise metrics
    train_accuracy = []
    round_durations = []
    round_energy = []

    clients, orbits = create_clients(train_dataset, num_clients=40)
    num_orbits = len(set(orbits))

    for rnd in range(args.epochs):
        print(f"\n--- Round {rnd + 1} ---")
        start_time = time.time()

        # Randomly select an orbit instead of using modulo
        selected_orbit = random.choice(list(set(orbits)))

        # Find all clients in the selected orbit
        orbit_clients = [i for i, orbit in enumerate(orbits) if orbit == selected_orbit]

        # Randomly choose one client from this orbit
        selected_client = random.choice(orbit_clients)

        print(f"  Selected Orbit: {selected_orbit}")
        print(f"  Clients in Orbit {selected_orbit}: {orbit_clients}")
        print(f"  Selected Client: {selected_client}")

        # Local training setup
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(global_model.state_dict())

        local_update_obj = LocalUpdate(args=args, dataset=train_dataset,
                                       idxs=user_groups[selected_client], logger=logger)
        updated_weights, loss = local_update_obj.update_weights(
                    model=copy.deepcopy(global_model), global_round=rnd)

        # Asynchronous aggregation
        global_model = async_aggregate(global_model, updated_weights, alpha=0.5)

        # Measure round duration and estimate energy
        duration = time.time() - start_time
        energy = estimate_energy_consumption(duration, device)

        round_durations.append(duration)
        round_energy.append(energy)

        # Evaluate accuracy
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
    total_time = sum(round_durations)
    total_energy = sum(round_energy)

    # Print final results
    print(f'\n Results after {args.epochs} global rounds:')
    print(f"|---- Avg Train Accuracy: {100 * train_accuracy[-1]:.2f}%")
    print(f"|---- Test Accuracy: {100 * test_acc:.2f}%")
    print(f"|---- Total Training Time: {total_time / 3600:.2f} hours")
    print(f"|---- Total Energy Consumption: {total_energy:.2f} units")

    return train_accuracy, test_acc, test_loss, total_time, total_energy
