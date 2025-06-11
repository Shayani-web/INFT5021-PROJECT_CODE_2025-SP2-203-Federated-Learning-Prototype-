

# To run all the algorithms, this file should be used and the code for execution is :
# python run_all.py

# Experimrntal setup as per "Jabbarpour et al. (2023). Performance Analysis of Federated Learning in Orbital Edge Computing. 
# In 2023 IEEE/ACM 16th International Conference on Utility and Cloud Computing (UCC’23). 
# https://doi.org/10.1145/3603166.3632140"

import subprocess

experiments = [
    {"model": "cnn", "dataset": "eurosat", "iid": "0"},
    {"model": "resnet18", "dataset": "eurosat", "iid": "0"},
    {"model": "cnn", "dataset": "cifar", "iid": "1"},
]

# Define which algorithms to run
runs = ["fedasync", "fedleo"]

# Define the number of local epochs
L_epochs =["10", "20", "30"]

# Define Batch Size
Batch_S=["8", "20", "32"]

# Loop through all combinations
for exp in experiments:
    for run in runs:
        for local_ep in L_epochs:
            for bs in Batch_S:
                print(f"\nRunning {run} with model={exp['model']}, dataset={exp['dataset']}, IID={exp['iid']}")
                subprocess.run([
                    "python", "src/federated_main.py",
                    "--model", exp["model"],
                    "--dataset", exp["dataset"],
                    "--iid", exp["iid"],    #IID/NON-IID data
                    "--epochs", "2",    #Define the number of global rounds
                    "--local_ep", local_ep,     # Define the number of local epochs
                    "--local_bs", bs,       # Batch size
                    "--run", run
                ])
