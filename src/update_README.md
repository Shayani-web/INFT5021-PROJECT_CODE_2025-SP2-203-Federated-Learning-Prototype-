# update.py

## Overview
The `update.py` file contains classes and functions for local training and inference in federated learning. It defines a `DatasetSplit` class for handling client-specific data subsets and a `LocalUpdate` class for performing local training and evaluation. The `test_inference` function evaluates the global model on the test dataset.

---

## Key Components and Functionality

### DatasetSplit Class

**Purpose:**  
Wraps a PyTorch dataset to provide a subset based on specified indices.

**Implementation:**
- Takes a dataset and a list of indices as input.
- Implements `__len__` (returns number of indices) and `__getitem__` (returns image and label for a given index).
- Converts images and labels to PyTorch tensors.

**Significance:**
- Enables efficient handling of client-specific data subsets in federated learning.

---

### LocalUpdate Class

**Purpose:**  
Manages local training and inference for a single client.

**Initialization:**
- Takes arguments, dataset, client indices, and a logger.
- Splits indices into training (80%), validation (10%), and test (10%) sets.
- Creates DataLoaders for each split with specified batch sizes.
- Uses `NLLLoss` as the default loss function.

**Methods:**

#### `train_val_test`:
- Splits indices and creates DataLoaders for training, validation, and testing.
- Training DataLoader uses shuffling; others do not.

#### `update_weights`:
- Trains the model for `args.local_ep` epochs using the specified optimizer (`sgd` or `adam`).
- Computes loss using `NLLLoss` and updates model weights via backpropagation.
- Logs loss to TensorBoard and prints verbose output for the first batch.



