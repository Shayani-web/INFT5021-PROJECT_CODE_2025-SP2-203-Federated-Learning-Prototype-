import os
import sys

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)

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
from FedAsync import fedAsync_Training
from paths import *
  # PLOTTING (optional)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context

def append_accuracy_csv(model_name, accuracy,specs, filename='accuracy_results.csv'):
    specs = ", ".join([spec for spec in specs])

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

    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(','.join(columns) + '\n')
    with open(filename, 'a') as f:
        f.write(f"{model_name}, {specs}, {accuracy}\n")
    print(f"Appended accuracy {accuracy} to {filename}.")

if __name__ == '__main__':
    start_time = time.time()
    
    # Setup logging and arguments
    logger = SummaryWriter(TENSORBOARD_LOG_DIR)
    args = args_parser()
    exp_details(args)
    
    
    # Load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    # Build model (extend for EuroSat if needed)
    if args.model == 'cnn' and args.dataset == 'cifar':
        global_model = CNNModel(dim_out=args.num_classes)

    elif args.model == 'cnn' and args.dataset == 'eurosat':
        global_model = EuroSATCNN(dim_out=args.num_classes)

    elif args.model == 'resnet18':
        global_model = ResNet18Model(dim_out=args.num_classes)

    
    # Run FL
    if args.run == 'fedleo':
        train_loss, train_accuracy, test_acc, total_time, total_energy = federated_learning(
            args, train_dataset, test_dataset, user_groups, global_model, logger)
    
        # Save results
        file_name = os.path.join(SAVE_DIR_PKL,f'{args.dataset}_{args.model}_{args.epochs}_hierarchical.pkl')
        with open(file_name, 'wb') as f:
            pickle.dump([train_loss, train_accuracy, test_acc, total_time, total_energy], f)
        
        print(f'\n Total Run Time: {time.time() - start_time:.4f} seconds')

        # PLOTTING (optional)
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')

        # Plot Loss curve
        plt.figure()
        plt.title('Training Loss vs Communication rounds')
        plt.plot(range(len(train_loss)), train_loss, color='r')
        plt.ylabel('Training loss')
        plt.xlabel('Communication Rounds')
        plt.savefig(os.path.join(SAVE_DIR, f'loss_{args.dataset}_{args.model}_{args.epochs}_hierarchical.png'))
        
        # Plot Average Accuracy vs Communication rounds
        plt.figure()
        plt.title('Average Accuracy vs Communication rounds')
        plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
        plt.ylabel('Average Accuracy')
        plt.xlabel('Communication Rounds')
        plt.savefig(os.path.join(SAVE_DIR, f'accuracy_{args.dataset}_{args.model}_{args.epochs}_hierarchical.png'))

    elif args.run == 'fedasync':
            train_accuracy, test_acc, test_loss = fedAsync_Training(
                args, train_dataset, test_dataset, user_groups, global_model, logger, device='cuda' if args.gpu else 'cpu')

            plt.figure()
            plt.title('Training Accuracy vs Communication rounds')
            plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
            plt.ylabel('Training Accuracy')
            plt.xlabel('Communication Rounds')
            plt.savefig(os.path.join(SAVE_DIR, f'train_accuracy_{args.dataset}_{args.model}_{args.epochs}_fedasync.png'))

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

    model_name = f"{args.run} for {args.dataset}"
    append_accuracy_csv(model_name, test_acc, specs)