import json
import os

import numpy as np

# Path to the main directory
base_dir = "results/cnn14/36/hubert"

# Strategies to loop over
strategies = ["smoothgrad"]

# Metrics to track
metrics = [
    "avg_test_loss",
    "avg_accuracy",
    "avg_f1",
    "avg_mask_mean",
    "avg_selective_accuracy",
    "avg_selective_f1",
]

# Iterate over each strategy
for strategy in strategies:
    strategy_path = os.path.join(base_dir, strategy)
    all_metrics = {metric: [] for metric in metrics}  # To collect metrics across folds

    # Iterate over each fold folder
    for fold_num in range(1, 11):
        fold_folder = os.path.join(strategy_path, f"fold_{fold_num}")
        results_path = os.path.join(fold_folder, "results.json")

        # Check if results.json exists
        if os.path.isfile(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
                # Collect each metric value
                for metric in metrics:
                    all_metrics[metric].append(results[metric])

    # Compute standard deviation for each metric across folds
    stddev_metrics = {
        metric: np.std(values) for metric, values in all_metrics.items() if values
    }

    # Print results for this strategy
    print(f"\nStandard deviation for strategy '{strategy}':")
    for metric, stddev in stddev_metrics.items():
        print(f"{metric}: {stddev:.6f}")
