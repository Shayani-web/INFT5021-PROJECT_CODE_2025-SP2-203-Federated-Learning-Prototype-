# federated_main.py

## Overview
The `federated_main.py` file serves as the entry point for running federated learning experiments, supporting both **FedLEO** (Federated Learning in Low Earth Orbits) and **FedAsync** (asynchronous federated learning). It handles:

- Dataset loading  
- Model initialisation  
- Training execution  
- Result saving  
- Visualisation of training metrics (loss and accuracy)  
- Logging results to a CSV file for easy comparison across experiments  

---

## Key Components and Functionality

### 1. Setup and Initialisation

**Purpose:**  
Configures the experiment by parsing arguments, setting up logging, and loading the dataset.

**Implementation:**
- Initialises a `SummaryWriter` from `tensorboardX` to log metrics to TensorBoard (`TENSORBOARD_LOG_DIR`).
- Parses command-line arguments using `args_parser` from `options.py`.
- Loads dataset and user groups using `get_dataset` from `utils.py`.
- Prints experiment configuration using `exp_details`.

---

### 2. Model Selection

**Purpose:**  
Initialises the global model based on the dataset and model type.

**Implementation:**
- For CIFAR-10 with `--model=cnn`, initiates `CNNModel` with output dimension `args.num_classes` (10 classes).
- For EuroSAT with `--model=cnn`, instantiates `EuroSATCNN`, tailored for 64x64 images.
- For `--model=resnet18`, instantiates `ResNet18Model` with a modified final layer for `args.num_classes`.
- Raises a `ValueError` for unsupported model-dataset combinations.

---

### 3. Federated Learning Execution

**Purpose:**  
Runs the selected federated learning algorithm (`fedleo` or `fedasync`).

**Implementation:**

#### If `--run=fedleo`:
- Calls `federated_learning` from `fedleo.py`.
- Saves results (loss, accuracy, test accuracy, time, energy) to a pickle file in `SAVE_DIR_PKL`.
- Plots and saves training loss and accuracy curves using Matplotlib.

#### If `--run=fedasync`:
- Calls `fedAsync_Training` from `FedAsyn.py`.
- Plots training accuracy.
- If FedLEO results are available, generates comparative plots for accuracy and loss.

**Outputs:**
- **FedLEO**: `train_loss`, `train_accuracy`, `test_acc`, `total_time`, `total_energy`
- **FedAsync**: `train_accuracy`, `test_acc`, `test_loss`, `total_time`, `total_energy`

---

### 4. Result Logging (`append_accuracy_csv`)

**Purpose:**  
Appends experiment results to `accuracy_results.csv`.

**Implementation:**
- Creates the CSV if it does not exist.
- Appends current experiment's:
  - Model name  
  - Model type  
  - Dataset  
  - Epochs  
  - Test accuracy  

**Significance:**  
Allows for tracking and comparing results across multiple experiments.

---

## Dependencies

### External Libraries
- `torch`
- `numpy`
- `tqdm`
- `tensorboardX`
- `matplotlib`

### Custom Modules
- `options.py`: Command-line argument parser
- `update.py`: Local training and inference logic
- `models.py`: Model architecture definitions
- `utils.py`: Dataset loading and utility functions
- `fedleo.py`: Implementation of FedLEO
- `FedAsyn.py`: Implementation of FedAsync
- `paths.py`: Directory path configuration

---


