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


P = 5800  # Orbital period in seconds (approx. for 600 km altitude) (equation for P is taken from the paper, pg.3.)
D = 600   # Visibility duration in seconds
N_ORBITS = 5    # Number of orbital planes in the constellation
SATS_PER_ORBIT = 8  # Number of satellites per orbital plane
TOTAL_SATELLITES = N_ORBITS * SATS_PER_ORBIT    # Total number of satellites (40)
T_TRAIN = 300  # Local training time in seconds (assumed 5 minutes)
ENERGY_ISL = 1.0  # Energy unit for ISL communication (relative)
ENERGY_GS = 10.0  # Energy unit for GS communication (relative)

# Compute phase offsets for each satellite
# The orbital period P is the time (in seconds) a satellite takes to complete one full orbit.
# We want to evenly distribute TOTAL_SATELLITES across this period,
# so they are spaced out in time (phase) along the orbit.
PHI = [s * (P / TOTAL_SATELLITES) for s in range(TOTAL_SATELLITES)]

# Define clusters (one per orbit)
# Each cluster contains SATS_PER_ORBIT satellites (e.g., satellites 0-7 in cluster 0)
CLUSTER_LIST = [list(range(i * SATS_PER_ORBIT, (i + 1) * SATS_PER_ORBIT)) for i in range(N_ORBITS)]

def next_visibility(s, t):
    """
    Calculate the next visibility start time for a satellite after a given time.

    This function determines when satellite `s` will next be visible to a ground
    station after time `t`, based on its phase offset and orbital period. The
    visibility time is computed as the smallest time after `t` when the satellite
    is in its visibility window.

    Args:
        s (int): Satellite index (0 to TOTAL_SATELLITES-1).
        t (float): Current time in seconds.

    Returns:
        float: Next visibility start time for satellite `s` in seconds.

    Mathematical Formulation:
        - Let PHI[s] be the phase offset of satellite s.
        - Orbital period P = 5800 seconds.
        - Number of full orbits elapsed: k = ceil((t - PHI[s]) / P).
        - Next visibility time: v_s = PHI[s] + k * P.
    """
    ## (t - PHI[s]) is the time elapsed since satellite `s` was last at its visibility point.
    k = math.ceil((t - PHI[s]) / P)
    # # Compute the actual time when satellite `s` becomes visible next.
    # This is its phase offset PHI[s] plus k, full orbital periods.
    v_s = PHI[s] + k * P
    return v_s

def federated_learning(args, train_dataset, test_dataset, user_groups, global_model, logger):
    """
    Implement the FedLEO federated learning algorithm with visibility-based timing and energy computation.

    FedLEO is a synchronous federated learning algorithm designed for LEO satellite constellations.
    It performs local training on each satellite, aggregates weights within orbital clusters,
    and then aggregates cluster weights globally after accounting for ground station visibility
    constraints. The algorithm tracks training loss, accuracy, time, and energy consumption.

    Args:
        args (argparse.Namespace): Command-line arguments containing hyperparameters
            (e.g., epochs, gpu, optimiser, learning rate).
        train_dataset (torch.utils.data.Dataset): Training dataset split among clients.
        test_dataset (torch.utils.data.Dataset): Test dataset for final evaluation.
        user_groups (dict): Dictionary mapping client indices to their data indices.
        global_model (torch.nn.Module): Global neural network model to be trained.
        logger (tensorboardX.SummaryWriter): Logger for TensorBoard metrics.

    Returns:
        tuple: Contains the following:
            - train_loss (list): Average training loss per global round.
            - train_accuracy (list): Average training accuracy per global round.
            - test_acc (float): Final test accuracy.
            - total_time (float): Total training time in seconds.
            - total_energy (float): Total energy consumption in relative units.

    Mathematical Formulations:
        - Cluster Aggregation: w_cluster = (1/|cluster|) * Σ(w_i), where w_i are local weights.
        - Global Aggregation: w_global = (1/|CLUSTER_LIST|) * Σ(w_cluster).
        - Energy per round: E_round = (TOTAL_SATELLITES * ENERGY_ISL) + (N_ORBITS * ENERGY_GS).
        - Training accuracy: acc = (1/num_users) * Σ(acc_i), where acc_i is client accuracy.
    """
    # Set device for computation (GPU if args.gpu is set, else CPU)
    device = 'cuda' if args.gpu else 'cpu'
    global_model.to(device)
    global_model.train()    # Set model to training mode
    
    # Initialise lists to track metrics across global rounds
    train_loss, train_accuracy = [], [] # Average training loss per round, Average training accuracy per round
    total_time = 0.0  # Cumulative training time in seconds
    total_energy = 0.0  # Cumulative energy consumption in relative units
    
    # Iterate over global rounds
    for epoch in tqdm(range(args.epochs), desc="Global Rounds"):
        print(f'\n | Global Training Round : {epoch + 1} |\n')
        # Initialise lists for cluster weights and local losses
        cluster_weights = []    # Aggregated weights for each cluster
        local_losses = []   # Training losses from local models
        # Record start time of the round
        t_start = total_time
        # End of local training time (start time + fixed training duration)
        t_train_end = t_start + T_TRAIN
        
        # Perform local training and aggregation within each cluster
        for cluster_idx, cluster in enumerate(CLUSTER_LIST):
            local_weights = []  # Store weights from satellites in the current cluster
            # Train each satellite (client) in the cluster
            for idx in cluster:
                # Initialise LocalUpdate for the client with its data subset
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger)
                # Perform local training, returning updated weights and loss
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))  # Store local weights
                local_losses.append(loss)   # Store local training loss
            # Aggregate weights within the cluster using average_weights
            # w_cluster = (1/|cluster|) * Σ(w_i)
            cluster_w = average_weights(local_weights)
            cluster_weights.append(cluster_w)
        
        # Determine master satellites and communication times
        v_c_list = []   # List of visibility times for master satellites
        for cluster in CLUSTER_LIST:
            # Select master as the satellite with earliest visibility after training
            s_master = min(cluster, key=lambda s: next_visibility(s, t_train_end))
            # Compute the next visibility time for the master satellite
            v_c = next_visibility(s_master, t_train_end)
            v_c_list.append(v_c)
        
        # Global aggregation occurs when all master satellites are visible
        # t_agg is the latest visibility time among all masters
        t_agg = max(v_c_list)
        total_time = t_agg
        
        # Perform global aggregation of cluster weights
        # w_global = (1/|CLUSTER_LIST|) * Σ(w_cluster)
        global_weights = average_weights(cluster_weights)
        global_model.load_state_dict(global_weights)
        
        # Compute energy consumption for the round
        # Energy includes ISL communication for all satellites and GS communication for each orbit
        energy_round = (TOTAL_SATELLITES * ENERGY_ISL) + (N_ORBITS * ENERGY_GS)
        total_energy += energy_round
        
        # Record average training loss for the round
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        # Evaluate training accuracy across all clients
        list_acc = []   # Store accuracy for each client
        global_model.eval() # Set model to evaluation mode
        for idx in range(args.num_users):
            # Initialise LocalUpdate for inference
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            # Compute accuracy on client's test split
            acc, _ = local_model.inference(model=global_model)
            list_acc.append(acc)
        # Average accuracy across all clients
        train_accuracy.append(sum(list_acc) / len(list_acc))
        
        # Print training statistics for the current round
        print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        print(f'Training Loss: {np.mean(np.array(train_loss)):.6f}')
        print(f'Train Accuracy: {100 * train_accuracy[-1]:.2f}%')
        print(f'Total Time: {total_time / 3600:.2f} hours')
        print(f'Total Energy: {total_energy:.2f} units')
    
    # Evaluate the final global model on the test dataset
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    
    # Print final results after all global rounds
    print(f'\n Results after {args.epochs} global rounds:')
    print(f"|---- Avg Train Accuracy: {100 * train_accuracy[-1]:.2f}%")
    print(f"|---- Test Accuracy: {100 * test_acc:.2f}%")
    print(f"|---- Total Training Time: {total_time / 3600:.2f} hours")
    print(f"|---- Total Energy Consumption: {total_energy:.2f} units")
    
     # Return metrics for further analysis or saving
    return train_loss, train_accuracy, test_acc, total_time, total_energy
