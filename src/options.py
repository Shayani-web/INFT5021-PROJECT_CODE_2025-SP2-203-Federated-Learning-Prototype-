#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    """
    Parse command-line arguments for configuring federated learning experiments.

    This function creates an ArgumentParser to handle hyperparameters and settings
    for federated learning algorithms (FedLEO and FedAsync), model architectures,
    datasets, and training configurations. The arguments control the experiment's
    behavior, such as the number of training rounds, client data distribution,
    model type, and optimization settings. The parsed arguments are used across
    the federated learning pipeline to ensure consistent configuration.

    Returns:
        argparse.Namespace: Parsed arguments containing experiment settings.

    Arguments:
        --epochs (int): Number of global training rounds (default: 10).
        --num_users (int): Number of clients/users (K) in the federated system (default: 100).
        --local_ep (int): Number of local epochs per client (E) (default: 10).
        --local_bs (int): Local batch size (B) for client training (default: 10).
        --lr (float): Learning rate for local training (default: 0.01).
        --momentum (float): Momentum for SGD optimizer (default: 0.5).
        --model (str): Model architecture ('cnn' or 'resnet18', default: 'resnet18').
        --kernel_num (int): Number of convolution kernels (default: 9, unused in current models).
        --kernel_sizes (str): Comma-separated kernel sizes for convolutions (default: '3,4,5', unused).
        --num_channels (int): Number of input image channels (default: 1, unused in current datasets).
        --norm (str): Normalization type ('batch_norm', 'layer_norm', or 'None', default: 'batch_norm', unused).
        --num_filters (int): Number of filters in conv nets (default: 32, unused).
        --max_pool (str): Whether to use max pooling ('True' or 'False', default: 'True', unused).
        --run (str): Federated learning algorithm ('fedleo' or 'fedasync', default: 'fedleo').
        --alpha (float): Blending factor for FedAsync aggregation (default: 0.5).
        --dataset (str): Dataset name ('cifar' or 'eurosat', default: 'eurosat').
        --num_classes (int): Number of classes in the dataset (default: 10).
        --gpu (str): GPU ID for CUDA (None for CPU, default: None).
        --optimizer (str): Optimizer type ('sgd' or 'adam', default: 'sgd').
        --iid (int): Data distribution (1 for IID, 0 for non-IID, default: 1).
        --unequal (int): Whether to use unequal data splits for non-IID (0 for equal, default: 0).
        --stopping_rounds (int): Rounds for early stopping (default: 10, unused).
        --verbose (int): Verbosity level (1 for verbose, default: 1).
        --seed (int): Random seed for reproducibility (default: 1).

    Notes:
        - Some arguments (e.g., kernel_num, kernel_sizes) are included for compatibility
          with other frameworks but are unused in the current implementation.
        - The default settings are tailored for the EuroSAT dataset with ResNet-18.
        - The --alpha argument is specific to FedAsync, controlling the weight of local updates.
    """

    # Initialize ArgumentParser for command-line argument parsing
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--run', type=str, default='fedleo',help= "federated model type"
    )
    parser.add_argument('--alpha', type=float, default=0.5, help="alpha value for FedAsync aggregation (default: 0.5)" 
    )
    
    parser.add_argument('--dataset', type=str, default='eurosat', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # Parse and return the arguments
    args = parser.parse_args()
    return args
