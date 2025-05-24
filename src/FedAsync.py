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


def create_clients(dataset, num_clients=40):  # Set the number of clients to 40 to match the setup used in the research paper
    indices = list(range(len(dataset)))  # create a list of all the indices in the dataset
    random.shuffle(indices)  # Then shuffle them to randomize how the data is split (so each client gets a mix)
    split_indices = np.array_split(indices, num_clients)  # Break the shuffled list into 40 chunks (one per client)

    # Each client gets a subset of the data using their portion of the indices
    return [Subset(dataset, idxs) for idxs in split_indices]



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


def evaluate_model(model, loader, device):
    model.eval()  # Set the model to evaluation mode (no dropout, no weight updates)

    all_preds, all_labels = [], []  # Store predictions and actual labels here

    with torch.no_grad():  # No need to compute gradients during evaluation (this saves memory and time)
        for batch in loader:
            inputs = batch["image"].to(device)   # Move the input images to the same device as the model (GPU/CPU)
            labels = batch["label"].to(device)   # Same for the labels

            outputs = model(inputs)              # Get predictions from the model
            _, preds = torch.max(outputs, 1)     # Get the index of the class with the highest score (i.e., the predicted label)

            # Bring predictions and labels back to the CPU and convert them to NumPy arrays
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # I calculate some evaluation metrics
    acc = accuracy_score(all_labels, all_preds)                 # How many predictions were correct?
    f1 = f1_score(all_labels, all_preds, average='macro')       # F1 Score balances precision and recall across all classes
    cm = confusion_matrix(all_labels, all_preds)                # Confusion matrix to identify which classes were confused

    # Return all three
    return acc, f1, cm


def fedAsync_Training(args, train_dataset, test_dataset, user_groups, global_model, logger, device):
    """Implement hierarchical FL with visibility-based timing and energy computation."""
    device = 'cuda' if args.gpu else 'cpu'
    global_model.to(device)
    global_model.train()
    
    # Initialize metrics
    train_accuracy = []
    num_rounds = 20
    local_epochs = 20 

    clients = create_clients(train_dataset, num_clients=40)

    # Loop through the number of training rounds
    for rnd in range(num_rounds):
        print(f"\n--- Round {rnd + 1} ---")  # track progress

        # Each round, simulates asynchronous behavior by picking one random client (instead of all clients training in sync)
        selected_client = random.choice(range(len(clients)))
        print(f"Selected Client: {selected_client}")  # Show which client was picked

        # Load that client's data into a DataLoader
        # Set batch size to 20 here (as used in the paper), and shuffle the data for randomness
        client_loader = DataLoader(clients[selected_client], batch_size=20, shuffle=True)

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