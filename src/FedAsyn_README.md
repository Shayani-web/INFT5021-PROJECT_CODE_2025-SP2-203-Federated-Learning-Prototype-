## FedAsyn.py – Asynchronous Federated Learning

The `FedAsyn.py` file implements the **FedAsync** algorithm, an asynchronous federated learning (FL) approach tailored for distributed training across multiple clients, especially relevant in space-based scenarios like satellite constellations.

Unlike synchronous FL, which requires all clients to synchronize their updates at fixed intervals, FedAsync allows clients to update the global model asynchronously, reducing waiting time and supporting intermittent connectivity [1].

---

### Key Components and Functionality

---

#### Client Creation and Orbit Assignment (`create_clients`)

**Purpose:**  
Divides the training dataset into subsets for multiple clients and assigns them to orbits to simulate a satellite constellation.

**Implementation Details:**
- Randomly shuffles dataset indices to avoid bias.
- Splits into `num_clients` (default: 40) subsets.
- Groups clients into `num_orbits` (default: 5) with `num_clients / num_orbits` clients per orbit.

---

#### Asynchronous Model Aggregation (`async_aggregate`)

**Purpose:**  
Updates the global model by blending the weights of a single client's local model with the current global model using a weighted average.

<<<<<<< HEAD
**Formula:**
<<<<<<< HEAD
$$
w_{\text{global}}[k] \leftarrow \alpha \cdot w_{\text{local}}[k] + (1 - \alpha) \cdot w_{\text{global}}[k]
$$
=======

![image](https://github.com/user-attachments/assets/6033fcca-98ba-422e-bc07-05c423391f87)

>>>>>>> 2760817809b0fb419ba2d80377942b04ca86abb3

- \( \alpha \in [0, 1] \) (default: 0.5) controls the influence of local updates.
- \( k \): layer or parameter index.

=======
>>>>>>> 1e0c7883bbb688e2d9d4d84098f20da702d15bc4
**Significance:**
- No need to wait for all clients.
- Efficient for real-time and power-sensitive systems.

---

#### Energy Consumption Estimation (`estimate_energy_consumption`)

**Purpose:**  
Estimates training round energy consumption based on device type and duration.

**Formula:**

![image](https://github.com/user-attachments/assets/69b1deca-4c0e-4f04-b4e4-a5c9289cbf46)

- \( P = 80W \) for GPU (`cuda`), \( 20W \) for CPU.
- \( t \): duration in seconds.

**Use Case:**  
Essential for evaluating power efficiency in satellite-based FL scenarios.

---

#### Main Training Loop (`fedAsync_Training`)

**Purpose:**  
Orchestrates the entire FedAsync training procedure.

**Key Steps:**
1. **Initialization:**
   - Set device (`cuda` or `cpu`)
   - Track metrics: accuracy, time, energy
   - Create clients and assign orbits

2. **Per Round Operations:**
   - Randomly select an orbit
   - Randomly pick one client from it
   - Train local model (via `LocalUpdate`)
   - Aggregate asynchronously (`async_aggregate`)
   - Record training duration and energy
   - Evaluate training accuracy across clients

3. **Final Evaluation:**
   - Evaluate the global model on the test set (`test_inference`)
   - Return key metrics

**Output:**
- Training accuracy per round
- Final test accuracy and loss
- Total training time
- Total energy consumption

---

### Dependencies

- **External:** `torch`, `numpy`, `tqdm`, `random`, `sklearn.metrics`
- **Internal:**
  - `options.py` – Argument parsing
  - `update.py` – Local training & inference
  - `utils.py` – Dataset loading and model utility functions

---

### Usage

Run this module by setting the `--run` argument to `fedasync` in `federated_main.py`.

```bash
python src/federated_main.py --run=fedasync --model=resnet18 --dataset=eurosat --epochs=20

