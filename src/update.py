#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """
    A custom Dataset class that wraps a PyTorch Dataset to provide a subset based on specified indices.

    This class enables federated learning by allowing each client to work with a specific subset
    of the dataset, defined by a list of indices. It converts dataset items (images and labels)
    to PyTorch tensors for compatibility with the training pipeline.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset to be split.
        idxs (list): List of indices defining the subset of the dataset.

    Attributes:
        dataset (torch.utils.data.Dataset): The original dataset.
        idxs (list): List of integer indices for the subset.

    Methods:
        __len__: Returns the number of items in the subset.
        __getitem__: Retrieves an item (image, label) from the dataset at the specified index.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        """Return the number of items in the subset."""
        return len(self.idxs)

    def __getitem__(self, item):
        """
        Retrieve an item from the dataset at the specified index and convert to tensors.

        Args:
            item (int): Index of the item to retrieve.

        Returns:
            tuple: (image, label) where both are PyTorch tensors.
        """
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    """
    A class to manage local training and inference for a single client in federated learning.

    This class handles the local training process for a client, including data splitting into
    train/validation/test sets, model training for multiple local epochs, and inference on the
    test set. It uses the specified optimizer and loss function (NLLLoss) to update the model
    weights and logs training metrics to TensorBoard.

    Args:
        args (argparse.Namespace): Command-line arguments (e.g., local_ep, local_bs, lr).
        dataset (torch.utils.data.Dataset): The dataset to be split for the client.
        idxs (list): List of indices defining the client's data subset.
        logger (tensorboardX.SummaryWriter): Logger for TensorBoard metrics.

    Attributes:
        args (argparse.Namespace): Experiment arguments.
        logger (tensorboardX.SummaryWriter): TensorBoard logger.
        trainloader (DataLoader): DataLoader for the training subset.
        validloader (DataLoader): DataLoader for the validation subset.
        testloader (DataLoader): DataLoader for the test subset.
        device (str): Device for computation ('cuda' or 'cpu').
        criterion (nn.Module): Loss function (NLLLoss).

    Methods:
        train_val_test: Splits indices into train/validation/test sets and creates DataLoaders.
        update_weights: Performs local training and returns updated model weights and loss.
        inference: Evaluates the model on the client's test set and returns accuracy and loss.
    """

    def __init__(self, args, dataset, idxs, logger):
        # Store arguments and logger
        self.args = args
        self.logger = logger
        # Create train, validation, and test DataLoaders for the client's data subset
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Split the client's data indices into train, validation, and test sets and create DataLoaders.

        The indices are split into 80% train, 10% validation, and 10% test. DataLoaders are created
        for each split with appropriate batch sizes and shuffling settings.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to be split.
            idxs (list): List of indices defining the client's data subset.

        Returns:
            tuple: (trainloader, validloader, testloader) containing DataLoader objects for
                   training, validation, and test sets.

        Notes:
            - Training DataLoader uses shuffling to improve training stability.
            - Validation and test DataLoaders use fixed batch sizes (1/10 of the set size) without shuffling.
        """
        # Split indices: 80% train, 10% validation, 10% test
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx == 0):
                    print('| Global Round : {} | Local Epoch : {} |\tLoss: {:.6f}'.format(
                        global_round, iter, loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """
    Evaluate the model on the client's test data subset and compute accuracy and loss.

    This method performs inference on the client's test set (self.testloader) using
    the provided model, calculating the total loss (using NLLLoss) and accuracy
    (correct predictions divided by total samples). The model is set to evaluation
    mode to disable gradient computation and dropout.

    Args:
        model (torch.nn.Module): The model to evaluate (typically the global model).

    Returns:
        tuple: (accuracy, total_loss)
            - accuracy (float): Fraction of correct predictions (correct/total).
            - total_loss (float): Sum of batch losses over the test set.

    Mathematical Formulation:
        - Loss: L_total = Σ(L_batch), where L_batch = NLLLoss(outputs, labels).
        - Accuracy: acc = (Σ(correct_predictions)) / total_samples, where
          correct_predictions = number of samples where predicted label equals true label.
    """
         # Set the model to evaluation mode (disables dropout and batch normalization updates)
        model.eval()
        # Initialize metrics for loss, correct predictions, and total samples
        loss, total, correct = 0.0, 0.0, 0.0

        # Iterate over batches in the test DataLoader
        for batch_idx, (images, labels) in enumerate(self.testloader):
            # Move data to the appropriate device (CPU or GPU)
            images, labels = images.to(self.device), labels.to(self.device)

            # Perform inference (forward pass) without gradient computation
            outputs = model(images)
            # Compute batch loss using Negative Log Likelihood Loss
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Compute predictions by selecting the class with the highest probability
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            # Count correct predictions by comparing with true labels
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            # Count total samples in the batch
            total += len(labels)

        # Calculate accuracy as the fraction of correct predictions
        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """
    Evaluate the global model on the entire test dataset and compute accuracy and loss.

    This function performs inference on the provided test dataset using the global model,
    calculating the total loss (using NLLLoss) and accuracy (correct predictions divided
    by total samples). It is used to evaluate the final performance of the global model
    after federated learning.

    Args:
        args (argparse.Namespace): Command-line arguments (e.g., gpu).
        model (torch.nn.Module): The global model to evaluate.
        test_dataset (torch.utils.data.Dataset): The test dataset.

    Returns:
        tuple: (accuracy, total_loss)
            - accuracy (float): Fraction of correct predictions (correct/total).
            - total_loss (float): Sum of batch losses over the test dataset.

    Mathematical Formulation:
        - Loss: L_total = Σ(L_batch), where L_batch = NLLLoss(outputs, labels).
        - Accuracy: acc = (Σ(correct_predictions)) / total_samples, where
          correct_predictions = number of samples where predicted label equals true label.
    """
    # Set the model to evaluation mode
    model.eval()
    # Initialize metrics for loss, correct predictions, and total samples
    loss, total, correct = 0.0, 0.0, 0.0

    # Set device based on GPU availability
    device = 'cuda' if args.gpu else 'cpu'
    # Initialize Negative Log Likelihood Loss
    criterion = nn.NLLLoss().to(device)
    # Create DataLoader for the test dataset with a fixed batch size
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    # Iterate over batches in the test DataLoader
    for batch_idx, (images, labels) in enumerate(testloader):
        # Move data to the appropriate device
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        # Compute batch loss
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Compute predictions by selecting the class with the highest probability
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        # Count correct predictions
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    # Calculate accuracy as the fraction of correct predictions
    accuracy = correct/total
    return accuracy, loss
