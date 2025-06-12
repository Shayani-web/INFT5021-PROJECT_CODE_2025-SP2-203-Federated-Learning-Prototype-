# ==== Standard Libraries ====
import os             # For interacting with the operating system (e.g., file paths, environment variables)
import time           # For measuring training duration and computing elapsed time
import math           # Provides access to mathematical functions like ceil, floor, etc.
import copy           # For making deep copies of models (to avoid reference-based updates)
import pickle         # For saving/loading Python objects (e.g., model weights, metrics)
import random         # For random selections (e.g., clients, orbits)

# ==== Scientific and Array Computation ====
import numpy as np    # Fundamental package for numerical computations and array manipulations

# ==== Progress Bar Utility ====
from tqdm import tqdm  # For displaying progress bars during loops or model training

# ==== PyTorch Core Modules ====
import torch                          # Main PyTorch library for tensor operations and GPU computation
import torch.nn as nn                 # Contains PyTorch's neural network layers and loss functions
import torch.optim as optim           # Optimisers like SGD, Adam for training models
from torch.utils.data import DataLoader, Dataset, random_split, Subset
# - DataLoader: for batch loading of datasets
# - Dataset: base class for custom datasets
# - random_split: splits datasets into training/validation
# - Subset: selects a subset of a dataset based on indices (used for client data partitioning)

# ==== Argument Parser and Project-Specific Modules ====
from options import args_parser             # Custom argument parser for managing CLI arguments
from update import LocalUpdate, test_inference
# - LocalUpdate: handles client-side training logic
# - test_inference: evaluates the global model on test data

from utils import get_dataset, average_weights, exp_details
# - get_dataset: loads and preprocesses datasets
# - average_weights: computes the average of weights from multiple clients
# - exp_details: prints/records experimental setup details

# ==== Evaluation Metrics ====
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# - accuracy_score: computes accuracy between predictions and true labels
# - f1_score: computes the F1-measure (harmonic mean of precision and recall)
# - confusion_matrix: shows classification performance in tabular format (TP, TN, FP, FN)


def create_clients(dataset, num_clients=40, num_orbits=5):
    """
    Splits the dataset among a specified number of clients and assigns clients to orbits.

    Args:
        dataset (Dataset): The full training dataset.
        num_clients (int): Total number of simulated clients. Default is 40.
        num_orbits (int): Number of orbital groups to assign clients into. Default is 5.

    Returns:
        clients (list): A list of `Subset` datasets, each representing one client's local data.
        orbits (list): A list where each index corresponds to a client and its assigned orbit.

    Effect:
        - Distributes the dataset evenly across clients.
        - Assigns each client to an orbit in a block-wise manner to simulate spatial grouping.
    """
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_indices = np.array_split(indices, num_clients)
    clients = [Subset(dataset, idxs) for idxs in split_indices]
    clients_per_orbit = num_clients // num_orbits
    orbits = [i for i in range(num_orbits) for _ in range(clients_per_orbit)]
    return clients, orbits


def async_aggregate(global_model, local_weights, alpha=0.5):
    """
    Performs asynchronous model aggregation using weighted blending of model parameters.

    Args:
        global_model (nn.Module): The current global model.
        local_weights (OrderedDict): Weights of the trained local model from a selected client.
        alpha (float): Mixing parameter (0 <= alpha <= 1). Controls the update intensity from the local model.
                       If alpha=0.5, both global and local models are weighted equally.

    Returns:
        global_model (nn.Module): The updated global model after asynchronous blending.

    Effect:
        - Applies the formula: new_weight = alpha * local_weight + (1 - alpha) * global_weight
        - Introduces partial updates without full synchronisation.
    """
    global_weights = global_model.state_dict()
    for key in global_weights:
        global_weights[key] = alpha * local_weights[key] + (1 - alpha) * global_weights[key]
    global_model.load_state_dict(global_weights)
    return global_model


def estimate_energy_consumption(duration_sec, device='cpu'):
    """
    Estimates energy consumed (in Joules) during a single training round.

    Args:
        duration_sec (float): Duration of the training round in seconds.
        device (str): Type of device used ('cpu' or 'cuda').

    Returns:
        energy_joules (float): Estimated energy usage in Joules.

    Effect:
        - Provides a simple energy usage metric using fixed average power draws.
    """
    power_draw_watts = 80 if device == 'cuda' else 20
    energy_joules = power_draw_watts * duration_sec
    return energy_joules

def fedAsync_Training(args, train_dataset, test_dataset, user_groups, global_model, logger, device):
    """
    Trains a global model using the FedAsyn (Asynchronous Federated Learning) protocol.

    In FedAsyn, only one randomly selected client (from a specific orbit/group) updates the global model in each round.
    This function also simulates orbit-based satellite client groupings and estimates energy consumption.

    Args:
        args (Namespace): Arguments containing FL configurations (e.g., number of users, epochs, GPU flag).
        train_dataset (Dataset): The complete dataset for local client training.
        test_dataset (Dataset): The dataset used for evaluating the final global model.
        user_groups (dict): Dictionary mapping client index to data indices.
        global_model (nn.Module): The base model shared with clients.
        logger (Logger): Logger for storing output logs (can be None).
        device (str): Computing device to be used ('cuda' or 'cpu').

    Returns:
        train_accuracy (list): List of average training accuracy after each round.
        test_acc (float): Final global model accuracy on the test dataset.
        test_loss (float): Final global model loss on the test dataset.
        total_time (float): Total training time in seconds.
        total_energy (float): Total estimated energy consumption in joules.

    Key Steps:
        1. Create clients and assign them into orbital groups.
        2. For each round:
            a. Select one orbit using a randomised but fair scheduling method.
            b. Pick one random client from that orbit.
            c. Perform local training for the client.
            d. Asynchronously blend the local model into the global model.
            e. Measure energy and time taken.
            f. Evaluate global model using inference across all clients.
        3. Evaluate final global model on held-out test dataset.
    """
    # Set device
    device = 'cuda' if args.gpu else 'cpu'
    global_model.to(device)
    global_model.train()

    # Initialise performance trackers
    train_accuracy = []
    round_durations = []
    round_energy = []

    # Step 1: Create clients and assign to orbits
    clients, orbits = create_clients(train_dataset, num_clients=40)
    num_orbits = len(set(orbits))  # Should equal args.num_orbits if controlled

    orbit_sequence = []  # Will hold shuffled orbit IDs for fair selection

    for rnd in range(args.epochs):
        print(f"\n--- Round {rnd + 1} ---")
        start_time = time.time()

        # Step 2a: Fair orbit selection (reshuffle every num_orbits rounds)
        if rnd % num_orbits == 0:
            orbit_sequence = random.sample(list(set(orbits)), num_orbits)

        selected_orbit = orbit_sequence[rnd % num_orbits]

        # Step 2b: Pick a random client from this orbit
        orbit_clients = [i for i, orbit in enumerate(orbits) if orbit == selected_orbit]
        selected_client = random.choice(orbit_clients)

        print(f"  Selected Orbit: {selected_orbit}")
        print(f"  Clients in Orbit {selected_orbit}: {orbit_clients}")
        print(f"  Selected Client: {selected_client}")

        # Step 2c: Initialize local update instance
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(global_model.state_dict())

        local_update_obj = LocalUpdate(args=args, dataset=train_dataset,
                                       idxs=user_groups[selected_client], logger=logger)
        updated_weights, loss = local_update_obj.update_weights(
            model=copy.deepcopy(global_model), global_round=rnd)

        # Step 2d: Asynchronous aggregation
        global_model = async_aggregate(global_model, updated_weights, alpha=0.5)

        # Step 2e: Time and energy tracking
        duration = time.time() - start_time
        energy = estimate_energy_consumption(duration, device)
        round_durations.append(duration)
        round_energy.append(energy)

        # Step 2f: Accuracy evaluation
        list_acc = []
        global_model.eval()
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[selected_client], logger=logger)
            acc, _ = local_model.inference(model=global_model)
            list_acc.append(acc)
        train_accuracy.append(sum(list_acc) / len(list_acc))

    # Step 3: Evaluate final global model on test set
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    # Final metrics
    total_time = sum(round_durations)
    total_energy = sum(round_energy)

    print(f'\n Results after {args.epochs} global rounds:')
    print(f"|---- Avg Train Accuracy: {100 * train_accuracy[-1]:.2f}%")
    print(f"|---- Test Accuracy: {100 * test_acc:.2f}%")
    print(f"|---- Total Training Time: {total_time / 3600:.2f} hours")
    print(f"|---- Total Energy Consumption: {total_energy:.2f} Joules")

    return train_accuracy, test_acc, test_loss, total_time, total_energy
