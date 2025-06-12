# Source Code Overview - FedLEO & FedAsyn

This document provides a technical breakdown of the Python source files used in the implementation of the **FedLEO** and **FedAsyn** federated learning algorithms. It supplements the main project-level README by focusing on architecture, parameters, and core logic.

---

## Core Scripts and What They Do

| File           | Description |
|----------------|-------------|
| `main.py`      | Entry point for training. Loads config, dataset, initialises models, and launches either FedAsyn or FedLEO based on `--run`. |
| `FedAsyn.py`   | Asynchronous FL implementation: picks a random client per round, blends weights with global model using `alpha`. |
| `fedleo.py`    | Hierarchical FL using clustered satellites, orbital phase scheduling, and energy accounting. |
| `models.py`    | Contains all CNN architectures: a basic CNN, ResNet18, and EuroSAT-custom CNN. |
| `update.py`    | Local client training: data splitting, weight updates, and inference logic. |
| `sampling.py`  | Implements both IID and Non-IID data partitioning strategies for CIFAR-10 and EuroSAT datasets. |
| `utils.py`     | Provides dataset loading (`get_dataset()`), weight averaging, and experiment logging helpers. |
| `paths.py`     | Automatically creates directories for logs and saved results. |
| `options.py`   | Argument parser with CLI flags for model type, dataset, learning rate, batch size, FL settings, and more. |

---

## Method-Specific Parameters

### FedAsyn

- `alpha`: controls weight blending between local and global model (`--alpha 0.5` default).
- Only one client is chosen per round.
- No satellite visibility scheduling, communication is immediate.

### FedLEO

- `P = 5800`: orbital period (s)
- `D = 600`: satellite visibility window (s)
- `N_ORBITS = 5`, `SATS_PER_ORBIT = 8`
- `T_TRAIN = 300`: local training duration (s)
- `ENERGY_ISL = 1.0`, `ENERGY_GS = 10.0`: relative energy cost of satellite-to-satellite and satellite-to-GS communication
- `PHI`: phase offsets used to simulate when satellites become visible
- Cluster formation is orbit-wise; one master satellite per cluster communicates with the GS

---

## Models

### `CNNModel` (for CIFAR-10)
-A basic Convolutional Neural Network (CNN) architecture.

-Contains 2 convolutional layers followed by 3 fully connected (dense) layers.

-Ends with a log_softmax layer for classification.

-Designed for CIFAR-10 images, which are 32×32 pixels with 3 colour channels (RGB).

-Lightweight and easy to train on standard image classification tasks.

### `EuroSATCNN`
-A modified CNN tailored for EuroSAT images, which are 64×64 RGB satellite images.

-The convolutional layers and fully connected layers are adjusted to handle the larger image size.

-Captures spatial features from satellite scenes like roads, forests, and rivers.

-Useful for testing FL performance on real-world satellite data.



### `ResNet18Model`
-Based on ResNet-18, a standard deep residual network from torchvision.models.

-Replaces the original final fully connected layer with a custom classifier suited to 10 classes for EuroSAT data.

-Can be used with both EuroSAT and CIFAR-10 for better accuracy due to its deeper architecture and skip connections.

-Suitable for more complex tasks and higher-performing FL experiments.


---

## Dataset Partitioning

| Dataset   | Sampling Method | Notes |
|-----------|------------------|-------|
| CIFAR-10  | IID & non-IID    | IID uses equal random split; non-IID assigns two shards per user |
| EuroSAT   | Only non-IID     | Each client gets data from 2 random classes, emulating satellite-specific scenes |

---

## Important Libraries

- `torch`, `torchvision`: for model, training, and data loading
- `numpy`, `random`: for shuffling, indexing, matrix ops
- `matplotlib`: to save plots of accuracy/loss curves
- `tensorboardX`: optional logging
- `argparse`: flexible runtime config
- `os`, `pickle`: for file handling and result serialisation

---

## Extending or Debugging

- To add a new dataset: modify `get_dataset()` in `utils.py` and add a sampling method to `sampling.py`.
- To use another model: define in `models.py`, update `main.py` selection logic.
- To change energy models in FedLEO: update constants in `fedleo.py` (e.g., `ENERGY_ISL`, `PHI` computation).

---

## Notes & Assumptions

- EuroSAT currently uses only RGB channels
- IID mode for EuroSAT is not implemented
- Phase scheduling and satellite visibility are simulated, not real-time

---

## Reference

This implementation is adapted from:

> Jabbarpour, M. R., Javadi, B., Leong, P. H. W., Calheiros, R. N., Boland, D., & Butler, C. (2023). Performance Analysis of Federated Learning in Orbital Edge Computing. In 2023 IEEE/ACM International Conference on Utility and Cloud Computing (UCC).(https://doi.org/10.1145/3603166.3632140)

---
