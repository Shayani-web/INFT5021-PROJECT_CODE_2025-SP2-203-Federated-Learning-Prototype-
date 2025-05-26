import subprocess

experiments = [
    {"model": "cnn", "dataset": "eurosat", "iid": "0"},
    {"model": "resnet18", "dataset": "eurosat", "iid": "0"},
    {"model": "cnn", "dataset": "cifar", "iid": "1"},
]

# Define which algorithms to run
runs = ["fedasync", "fedleo"]

# Loop through all combinations
for exp in experiments:
    for run in runs:
        print(f"\nRunning {run} with model={exp['model']}, dataset={exp['dataset']}, IID={exp['iid']}")
        subprocess.run([
            "python", "src/federated_main.py",
            "--model", exp["model"],
            "--dataset", exp["dataset"],
            "--iid", exp["iid"],
            "--epochs", "1",
            "--run", run
        ])
