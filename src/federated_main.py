import os
import sys

# Determine the main project directory and add it to the system path
main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)   # Allows importing custom modules from the project root

import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import CNNModel, ResNet18Model, EuroSATCNN
from utils import get_dataset, average_weights, exp_details
from fedleo import federated_learning
from FedAsyn import fedAsync_Training
from paths import *
  # PLOTTING (optional)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context

def append_accuracy_csv(model_name, accuracy,specs, filename='accuracy_results.csv'):
    """
    Append experiment results to a CSV file for tracking and comparison.

    This function logs the model name, experiment specifications, and test accuracy
    to a CSV file. If the file does not exist, it creates it with appropriate headers.
    The specifications include model type, dataset, epochs, and other hyperparameters.

    Args:
        model_name (str): Name of the model or experiment (e.g., 'fedleo for eurosat').
        accuracy (float): Test accuracy of the model.
        specs (list): List of experiment specifications (model, dataset, epochs, etc.).
        filename (str, optional): Name of the CSV file. Defaults to 'accuracy_results.csv'.

    Returns:
        None

    Notes:
        - The CSV file is created in the current working directory if it does not exist.
        - Columns include: Model Name, Model, Dataset, Epochs, Local Epochs, Local Batch Size,
          Learning Rate, Optimiser, IID, Accuracy.
        - The function prints a confirmation message after appending the results.
    """
    # Format specifications as a comma-separated string
    specs = ", ".join([spec for spec in specs])

    # Define CSV column headers
    columns = [
        "Model Name",
        "Model",
        "Dataset",
        "Epochs",
        "Local Epochs",
        "Local Batch Size",
        "Learning Rate",
        "Optimizer",
        "IID",
        "Accuracy"
    ]

    # Create CSV file with headers if it does not exist
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(','.join(columns) + '\n')
    # Append experiment results to the CSV file
    with open(filename, 'a') as f:
        f.write(f"{model_name}, {specs}, {accuracy}\n")
    print(f"Appended accuracy {accuracy} to {filename}.")

if __name__ == '__main__':
    # Record the start time of the experiment
    start_time = time.time()
    
    # Initialise TensorBoard logger for tracking metrics
    logger = SummaryWriter(TENSORBOARD_LOG_DIR)

    # Parse command-line arguments to configure the experiment
    args = args_parser()

    # Print experiment details (model, optimiser, dataset, etc.)
    exp_details(args)
    
    
    # Load training and test datasets, along with user groups (client data splits)
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    # Initialise the global model based on dataset and model type
    if args.model == 'cnn' and args.dataset == 'cifar':
        global_model = CNNModel(dim_out=args.num_classes)

    elif args.model == 'cnn' and args.dataset == 'eurosat':
        global_model = EuroSATCNN(dim_out=args.num_classes)

    elif args.model == 'resnet18':
        global_model = ResNet18Model(dim_out=args.num_classes)

    
    # Run the specified federated learning algorithm
    if args.run == 'fedleo':
        train_loss, train_accuracy, test_acc, total_time, total_energy = federated_learning(
            args, train_dataset, test_dataset, user_groups, global_model, logger)
    
        # Save results
        file_name = os.path.join(SAVE_DIR_PKL,f'{args.dataset}_{args.model}_{args.epochs}_hierarchical.pkl')
        with open(file_name, 'wb') as f:
            pickle.dump([train_loss, train_accuracy, test_acc, total_time, total_energy], f)
        
        print(f'\n Total Run Time: {time.time() - start_time:.4f} seconds')


    elif args.run == 'fedasync':  #Run FedAsync
            train_accuracy, test_acc, test_loss, total_time, total_energy = fedAsync_Training(
                args, train_dataset, test_dataset, user_groups, global_model, logger, device='cuda' if args.gpu else 'cpu')


            # Load previously saved FedLEO results for comparison (if exists)
            fedleo_path = os.path.join(SAVE_DIR_PKL, f'{args.dataset}_{args.model}_{args.epochs}_hierarchical.pkl')
            if os.path.exists(fedleo_path):
                with open(fedleo_path, 'rb') as f:
                    fedleo_loss, fedleo_accuracy, _, _, _ = pickle.load(f)

    # Prepare experiment specifications for CSV logging
    specs = [
        f"{args.model}",
        f"{args.dataset}",
        f"{args.epochs}",
        f"{args.local_ep}",
        f"{args.local_bs}",
        f"{args.lr}",
        f"{args.optimizer}",
        f"{args.iid}",
    ]

    # Log results to CSV
    model_name = f"{args.run} for {args.dataset}"
    append_accuracy_csv(model_name, test_acc, specs)
