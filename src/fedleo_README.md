# fedleo.py

## Overview

The fedleo.py file implements the FedLEO algorithm, a synchronous federated learning approach tailored for Low Earth Orbit (LEO) satellite constellations. It accounts for satellite visibility windows and communication constraints, simulating a realistic scenario where satellites communicate model updates via inter-satellite links (ISL) and ground stations (GS). The algorithm aggregates model weights within orbital clusters before performing global aggregation, and it tracks energy consumption and training time.

---

## Key Components and Functionality

### 1. Constants and Configuration

**Purpose:** Define parameters for satellite orbits and energy model.

**Constants:**
- `P = 5800` seconds: Orbital period for a 600 km altitude satellite.
- `D = 600` seconds: Duration of visibility with ground stations.
- `N_ORBITS = 5`: Number of orbital planes.
- `SATS_PER_ORBIT = 8`: Satellites per orbit, totaling `TOTAL_SATELLITES = 40`.
- `T_TRAIN = 300` seconds: Local training time per round (5 minutes) [1], a fixed estimate.
- `ENERGY_ISL = 1.0`: Relative energy cost of inter-satellite communication.
- `ENERGY_GS = 10.0`: Relative energy cost of ground station communication.

---

---
### 1. Orbital Period
Each satellite in each orbit travels at the same speed, ![image](https://github.com/user-attachments/assets/6f40e5e4-8f3b-40cc-b460-0b5328cb2e32)
 and has the same orbital period, ![image](https://github.com/user-attachments/assets/61a00815-b319-4870-ba80-13cd07736239)
. Here,
- ![image](https://github.com/user-attachments/assets/7e926db0-a740-4512-91bf-6c09d9e329bb) -------------------------------- [1]

and

- ![image](https://github.com/user-attachments/assets/fd410cb4-8c4b-43fe-8593-db85662c934f) --------------------------------- [1]

where ![image](https://github.com/user-attachments/assets/4c139487-14bb-4a04-b43e-ea59e17f92a3) is the earth radius (6371km), ![image](https://github.com/user-attachments/assets/1337c0c5-8e73-4cf3-9c55-10df94fb7cba) is geocentric gravitational constant ![image](https://github.com/user-attachments/assets/0fd7f82e-bfd8-45e9-b1bf-58812b499af8) and ![image](https://github.com/user-attachments/assets/8ca8d461-11c0-4e16-a4ae-72d67ecf6876) is orbit altitude (600km)



### 3. Phase Offsets (PHI)

Satellites are evenly spaced across their orbital period:

![image](https://github.com/user-attachments/assets/247aae46-48f1-43e4-af69-a0ccaba83518)


The phase offset is calculated to ensure even distribution of satellites across the orbital period.

---

### 4. Cluster Definition (`CLUSTER_LIST`)

Satellites are grouped into clusters (one per orbit), each containing `SATS_PER_ORBIT` satellites. Each cluster performs localized aggregation before sending a representative update.

---

### 5. Next Visibility Calculation (`next_visibility`)

**Purpose:**  
Compute when a satellite will next be visible to a ground station.

**Formula:**

- ![image](https://github.com/user-attachments/assets/3abf0b3c-11d8-45bb-ba92-3c47bc6a706e)

- ![image](https://github.com/user-attachments/assets/cbf41a48-da3f-4771-a85e-0046ddbae140)





Where:
- ![image](https://github.com/user-attachments/assets/21c582d6-b244-4107-8853-3044df17e1cf)
 is the phase offset
- ![image](https://github.com/user-attachments/assets/43b814b8-ab99-487d-9315-175ec8aa1732)
 is the orbital period
- ![image](https://github.com/user-attachments/assets/9ba548ce-4a72-4ebb-934d-63a9b8fd0eb7)
 is the next visibility time

---

### 6. Main Training Loop (`federated_learning`)

**Purpose:**  
Runs the FedLEO algorithm over multiple global rounds.

**Key Steps:**

#### Initialization:
- Load global model to device.
- Prepare lists for tracking metrics.

#### Local Training:
- Each client trains using `LocalUpdate`.
- Local weights and losses are collected.

#### Cluster Aggregation:
![image](https://github.com/user-attachments/assets/3dcfcb18-635b-4b5a-aff2-98614f8054fd)


#### Master Selection and Communication:
- Master node is the one with the **earliest visibility**.
- Aggregation time:
![image](https://github.com/user-attachments/assets/7bfe1670-efdc-4829-b6df-3075813303ea)


#### Global Aggregation:
Aggregates cluster weights globally:
![image](https://github.com/user-attachments/assets/ed1332fa-427f-42d9-9337-02df3fa7643b)

-Updates the global model with `load_state_dict`.


#### Energy Calculation:
![image](https://github.com/user-attachments/assets/d9870041-9b11-424d-b895-4b46b7b5a428)


#### Evaluation:
- Computes loss and accuracy across all clients.
- Final test accuracy is calculated post training.


# Federated Learning Function (federated_learning)

## Purpose
Orchestrates the FedLEO training process, integrating local training, cluster aggregation, visibility-based synchronization, and metric tracking.

## Implementation

### Initialization:
- Moves the global model to the specified device (GPU if available, otherwise CPU).
- Initializes lists to track training loss, accuracy, total time, and energy consumption.

### Per Global Round (for `args.epochs` rounds):

#### Local Training and Cluster Aggregation:
- For each cluster (representing an orbit), trains all satellites using the `LocalUpdate.update_weights` method from `update.py`.
- Aggregates the local weights within each cluster to produce a single set of weights per cluster.
- Stores these cluster weights for further processing.

#### Master Selection and Visibility Timing:
- Selects the master satellite for each cluster, defined as the one with the earliest visibility time after local training ends.
- Determines the visibility times for all master satellites and sets the global aggregation time as the latest of these times.

#### Global Aggregation:
- Combines the weights from all clusters into a single global weight set.
- Updates the global model with these aggregated weights.

#### Energy Consumption:
- Estimates the energy used in the round by accounting for communication costs between all satellites and with ground stations.
- Adds this energy cost to the running total.

#### Metrics Collection:
- Computes the average training loss and accuracy by evaluating the model across all clients using the `LocalUpdate.inference` method.
- Updates the respective tracking lists with these metrics.

#### Logging:
- Prints statistics for the current round, including loss, accuracy, total time, and energy, and provides a final summary after all rounds.

### Final Evaluation:
- Assesses the global model’s performance on the test dataset using the `test_inference` method from `update.py`.
- Returns the training loss history, training accuracy history, test accuracy, total time, and total energy for further analysis.


---

### Outputs
Returns:
- `train_loss`
- `train_accuracy`
- `test_acc`
- `total_time`
- `total_energy`

---

## Dependencies

### External Libraries:
- `torch`
- `numpy`
- `tqdm`

### Custom Modules:
- `options.py`: Argument parsing
- `update.py`: Local training and testing logic
- `utils.py`: Dataset loading and model utilities

---

## Usage

This file is called from federated_main.py when --run=fedleo. It requires a dataset, model, and user groups, and it simulates a satellite-based federated learning scenario with visibility constraints.

---
## References
> 1. Jabbarpour, M. R., Javadi, B., Leong, P. H. W., Calheiros, R. N., Boland, D., & Butler, C. (2023). Performance Analysis of Federated Learning in Orbital Edge Computing. In 2023 IEEE/ACM International Conference on Utility and Cloud Computing (UCC).(https://doi.org/10.1145/3603166.3632140)
---

