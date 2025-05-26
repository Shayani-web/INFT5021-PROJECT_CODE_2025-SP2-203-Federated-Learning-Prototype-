# Federated Learning for Orbital Edge Computing: FedLEO vs FedAsync

This repository reproduces and compares two Federated Learning (FL) algorithms, **FedAsync** and **FedLEO** in the context of Orbital Edge Computing (OEC). The implementation is guided by the research presented in:

**Jabbarpour et al. (2023).** *Performance Analysis of Federated Learning in Orbital Edge Computing*. In *2023 IEEE/ACM 16th International Conference on Utility and Cloud Computing (UCC’23)*. [https://doi.org/10.1145/3603166.3632140](https://doi.org/10.1145/3603166.3632140)

---

## Overview

The emergence of Low Earth Orbit (LEO) satellite constellations presents new opportunities for edge computing in space. Instead of relying solely on ground stations, **Orbital Edge Computing (OEC)** enables satellites to process data in orbit. Federated Learning fits this model well, as data can remain local to each satellite.

This project implements and evaluates:

- **FedAsync**: Centralised, asynchronous aggregation of client updates.
- **FedLEO**: Decentralised, hierarchical training using intra-orbit clusters.

These approaches are tested on:

- **CIFAR-10 (IID data)** using a CNN model
- **EuroSAT (Non-IID satellite imagery)** using ResNet18

---

## Algorithm Descriptions

### FedAsync (Asynchronous Federated Learning)

- Only one random client is selected per round.
- The global model is updated using the selected client's weights with a mixing parameter α (e.g., 0.5).
- No global synchronisation is needed each round.

**Pros**: Saves energy and training time  
**Cons**: Accuracy can fluctuate due to stale updates

> “Although the asynchronous centralised FL method has high fluctuations in the accuracy curve, it is suitable for space applications in which power consumption and training time are two main factors.”  
> -*Jabbarpour et al., 2023*

---

### FedLEO (Federated Learning in Low Earth Orbit)

- Satellites are grouped by orbits into clusters.
- Local training is followed by intra-cluster aggregation.
- One satellite per cluster communicates with the Ground Station (GS).
- Aggregation timing is calculated based on visibility patterns.

**Pros**: Efficient in non-IID environments; lower GS communication load  
**Cons**: Requires synchronisation and timing models

> “FedLEO improves model propagation through intra-plane communication and leverages the predictability of satellite orbiting patterns.”  
> -*Jabbarpour et al., 2023*

---

## Experimental Setup

| Parameter   | Value                             |
|-------------|-----------------------------------|
| Satellites  | 40 (5 orbits × 8 satellites)      |
| Datasets    | CIFAR-10 (IID), EuroSAT (Non-IID) |
| Models      | CNN, ResNet18                     |
| Metrics     | Accuracy, Training Time, Power    |

## Key Findings

- Higher training rounds boost accuracy but increase energy and time.
- FedAsync is well-suited for power-limited environments.
- FedLEO yields better accuracy for non-IID data, particularly on EuroSAT.
- Increasing satellite altitude reduces training time but increases communication energy.
- Larger batch sizes and more sampled satellites improve learning in non-IID conditions.

> “The number of sampled satellites (cluster size) is a more important parameter in non-IID data distribution than in IID.”  
> -*Jabbarpour et al., 2023*

---

## Citation

If you use or extend this work, please cite the original paper:

**APA:**

Jabbarpour, M. R., Javadi, B., Leong, P. H. W., Calheiros, R. N., Boland, D., & Butler, C. (2023). *Performance Analysis of Federated Learning in Orbital Edge Computing*. In *2023 IEEE/ACM International Conference on Utility and Cloud Computing (UCC)*. [https://doi.org/10.1145/3603166.3632140](https://doi.org/10.1145/3603166.3632140)

### Directory Structure
.

├── data/                    # Input datasets

├── save/                    # Model outputs, plots, and results

├── src/

│   ├── [FedAsync.py](src/FedAsync.py)          # FedAsync training logic

│   ├── [fedleo.py](src/fedleo.py)              # FedLEO training logic with timing & energy model

│   ├── [federated_main.py](src/federated_main.py)  # Main script to launch training

│   ├── [models.py](src/models.py)              # CNN, ResNet18, EuroSATCNN definitions

│   ├── [options.py](src/options.py)            # Command Line Interface argument parsing

│   ├── [paths.py](src/paths.py)                # Path configurations

│   ├── [sampling.py](src/sampling.py)          # IID and Non-IID partitioning on the datasets

│   ├── [update.py](src/update.py)              # Local train/test logic

│   └── [utils.py](src/utils.py)                # Weight averaging, loaders, helpers

├── [requirments.txt](requirments.txt)          # Python dependencies

└── [.gitignore](.gitignore)                    # Specifies files and folders to be ignored by Git
