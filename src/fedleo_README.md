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
- `T_TRAIN = 300` seconds: Local training duration.
- `ENERGY_ISL = 1.0`: Relative energy cost of inter-satellite communication.
- `ENERGY_GS = 10.0`: Relative energy cost of ground station communication.

---

### 2. Phase Offsets (PHI)

Satellites are evenly spaced across their orbital period:

\[
\phi_s = s \cdot \frac{P}{\text{TOTAL\_SATELLITES}}, \quad s = 0, 1, \ldots, \text{TOTAL\_SATELLITES} - 1
\]

This ensures staggered satellite visibility for realistic simulation.

---

### 3. Cluster Definition (`CLUSTER_LIST`)

Satellites are grouped into clusters (one per orbit), each containing `SATS_PER_ORBIT` satellites. Each cluster performs localized aggregation before sending a representative update.

---

### 4. Next Visibility Calculation (`next_visibility`)

**Purpose:**  
Compute when a satellite will next be visible to a ground station.

**Formula:**

\[
k = \left\lceil \frac{t - \phi_s}{P} \right\rceil
\quad \Rightarrow \quad
v_s = \phi_s + k \cdot P
\]

Where:
- \( \phi_s \) is the phase offset
- \( P \) is the orbital period
- \( v_s \) is the next visibility time

---

### 5. Main Training Loop (`federated_learning`)

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
\[
w_{\text{cluster}} = \frac{1}{|\text{cluster}|} \sum_{i \in \text{cluster}} w_i
\]

#### Master Selection and Communication:
- Master node is the one with the **earliest visibility**.
- Aggregation time:
\[
t_{\text{agg}} = \max_{c} v_c
\]

#### Global Aggregation:
\[
w_{\text{global}} = \frac{1}{|\text{CLUSTER\_LIST}|} \sum_{c} w_{\text{cluster}_c}
\]

#### Energy Calculation:
\[
E_{\text{round}} = (\text{TOTAL\_SATELLITES} \cdot \text{ENERGY\_ISL}) + (\text{N\_ORBITS} \cdot \text{ENERGY\_GS})
\]

#### Evaluation:
- Computes loss and accuracy across all clients.
- Final test accuracy is calculated post training.

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



