# options.py

## Overview
The `options.py` file provides a command-line argument parser for configuring federated learning experiments. It defines hyperparameters for the federated learning setup, model architecture, dataset, and training process, allowing flexible experimentation with different settings.

---

## Key Components and Functionality

### Argument Parser (`args_parser`)

**Purpose:**  
Parses command-line arguments to configure the experiment.

**Arguments:**

#### Federated Learning Parameters:
- `--epochs`: Number of global training rounds (default: 10).
- `--num_users`: Number of clients (default: 100).
- `--local_ep`: Number of local epochs per client (default: 10).
- `--local_bs`: Local batch size (default: 10).
- `--lr`: Learning rate (default: 0.01).
- `--momentum`: SGD momentum (default: 0.5).

#### Model Parameters:
- `--model`: Model type (cnn or resnet18, default: resnet18).
- `--kernel_num`, `--kernel_sizes`, `--num_channels`, `--num_filters`, `--norm`, `--max_pool`: CNN-specific parameters.

#### Other Parameters:
- `--run`: Federated learning algorithm (fedleo or fedasync, default: fedleo).
- `--alpha`: Blending factor for FedAsync aggregation (default: 0.5).
- `--dataset`: Dataset name (cifar or eurosat, default: eurosat).
- `--num_classes`: Number of classes (default: 10).
- `--gpu`: GPU ID (default: None for CPU).
- `--optimizer`: Optimiser type (sgd or adam, default: sgd).
- `--iid`: Whether data is IID (1 for IID, 0 for non-IID, default: 1).
- `--unequal`: Whether to use unequal data splits (default: 0).
- `--stopping_rounds`: Early stopping rounds (default: 10, unused).
- `--verbose`: Verbosity level (default: 1).
- `--seed`: Random seed (default: 1).

---

## Dependencies

### External Libraries:
- `argparse`

---

## Usage

The `args_parser` function is called in `federated_main.py` to parse command-line arguments:

```bash
python src\federated_main.py --run=fedasync --dataset=eurosat --model=cnn --epochs=10 --num_users=40

