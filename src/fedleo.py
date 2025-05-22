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


P = 5800  # Orbital period in seconds (approx. for 600 km altitude)
D = 600   # Visibility duration in seconds
N_ORBITS = 5
SATS_PER_ORBIT = 8
TOTAL_SATELLITES = N_ORBITS * SATS_PER_ORBIT
T_TRAIN = 300  # Local training time in seconds (assumed 5 minutes)
ENERGY_ISL = 1.0  # Energy unit for ISL communication (relative)
ENERGY_GS = 10.0  # Energy unit for GS communication (relative)

# Compute phase offsets for each satellite
# The orbital period P is the time (in seconds) a satellite takes to complete one full orbit.
# We want to evenly distribute TOTAL_SATELLITES across this period,
# so they are spaced out in time (phase) along the orbit.
PHI = [s * (P / TOTAL_SATELLITES) for s in range(TOTAL_SATELLITES)]

# Define clusters (one per orbit)
CLUSTER_LIST = [list(range(i * SATS_PER_ORBIT, (i + 1) * SATS_PER_ORBIT)) for i in range(N_ORBITS)]

def next_visibility(s, t):
    """Calculate the next visibility start time for satellite s after time t."""
    ## (t - PHI[s]) is the time elapsed since satellite `s` was last at its visibility point.
    k = math.ceil((t - PHI[s]) / P)
    # # Compute the actual time when satellite s becomes visible next.
    # This is its phase offset PHI[s] plus k, full orbital periods.
    v_s = PHI[s] + k * P
    return v_s

def federated_learning(args, train_dataset, test_dataset, user_groups, global_model, logger):
    """Implement federated learning with visibility-based timing and energy computation."""
    device = 'cuda' if args.gpu else 'cpu'
    global_model.to(device)
    global_model.train()
    
    # Initialize metrics
    train_loss, train_accuracy = [], []
    total_time = 0.0  # Cumulative training time in seconds
    total_energy = 0.0  # Cumulative energy consumption
    
    for epoch in tqdm(range(args.epochs), desc="Global Rounds"):
        print(f'\n | Global Training Round : {epoch + 1} |\n')
        cluster_weights = []
        local_losses = []
        t_start = total_time
        t_train_end = t_start + T_TRAIN
        
        # Local training and aggregation within each cluster
        for cluster_idx, cluster in enumerate(CLUSTER_LIST):
            local_weights = []
            for idx in cluster:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger)
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(loss)
            # Aggregate weights within the cluster
            cluster_w = average_weights(local_weights)
            cluster_weights.append(cluster_w)
        
        # Determine master satellites and communication times
        v_c_list = []
        for cluster in CLUSTER_LIST:
            # Select master as the satellite with earliest visibility after training
            s_master = min(cluster, key=lambda s: next_visibility(s, t_train_end))
            v_c = next_visibility(s_master, t_train_end)
            v_c_list.append(v_c)
        
        # Global aggregation occurs when all masters have communicated
        t_agg = max(v_c_list)
        total_time = t_agg
        
        # Perform global aggregation
        global_weights = average_weights(cluster_weights)
        global_model.load_state_dict(global_weights)
        
        # Compute energy consumption for this round
        energy_round = (TOTAL_SATELLITES * ENERGY_ISL) + (N_ORBITS * ENERGY_GS)
        total_energy += energy_round
        
        # Record training loss
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        # Evaluate training accuracy
        list_acc = []
        global_model.eval()
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, _ = local_model.inference(model=global_model)
            list_acc.append(acc)
        train_accuracy.append(sum(list_acc) / len(list_acc))
        
        print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        print(f'Training Loss: {np.mean(np.array(train_loss)):.6f}')
        print(f'Train Accuracy: {100 * train_accuracy[-1]:.2f}%')
        print(f'Total Time: {total_time / 3600:.2f} hours')
        print(f'Total Energy: {total_energy:.2f} units')
    
    # Test the final model
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    
    # Print final results
    print(f'\n Results after {args.epochs} global rounds:')
    print(f"|---- Avg Train Accuracy: {100 * train_accuracy[-1]:.2f}%")
    print(f"|---- Test Accuracy: {100 * test_acc:.2f}%")
    print(f"|---- Total Training Time: {total_time / 3600:.2f} hours")
    print(f"|---- Total Energy Consumption: {total_energy:.2f} units")
    
    return train_loss, train_accuracy, test_acc, total_time, total_energy