import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datasets.audio_classification_dataset import AudioClassificationDataset
from models.ssl_classification_model import SSLClassificationModel
import pandas as pd
from yaml_config_override import add_arguments
from addict import Dict
import argparse
import numpy as np 

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set all seeds to {seed}")

def load_model_checkpoint(checkpoint_path, config, device):
    model = SSLClassificationModel(config=config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def eval_one_epoch(model, eval_dataloader, device, is_binary_classification=False):
    model.eval()
    eval_loss = 0.0
    reference = []
    predictions = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), ncols=100):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(batch)
            n_classes = outputs.shape[-1]
            if is_binary_classification:
                predictions.extend((outputs > 0.5).cpu().numpy().astype(int))
            else:
                predictions.extend(torch.argmax(outputs, dim=-1).cpu().numpy().astype(int))
            reference.extend(labels.cpu().numpy())
    return reference, predictions

def compute_metrics(reference, predictions, is_binary_classification=False):
    accuracy = accuracy_score(reference, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(reference, predictions, average="macro")
    cm = confusion_matrix(reference, predictions)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }
    return metrics

def get_validation_dataloader(val_path, class_mapping, config):
    df_val = pd.read_csv(val_path)
    val_paths, val_labels = df_val.audio_path.values.tolist(), df_val[config.training.label_key].values.tolist()
    val_ds = AudioClassificationDataset(
        audio_paths=val_paths,
        labels=val_labels,
        feature_extractor_name_or_path=config.model.model_name_or_path,
        class_mapping=class_mapping,
        data_config=config.data,
        is_test=True
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    return val_dl
import os

def get_checkpoint_and_val_paths(config, fold_num):
    """Get the appropriate checkpoint and validation paths based on the fold number and model name."""
    base_dir = os.getcwd()  # Get the current working directory

    if config.model.model_name_or_path == "facebook/hubert-base-ls960":
        checkpoint_path = os.path.join(base_dir, "checkpoints", "SSL4PR-hubert-base", f"fold_{fold_num}.pt")
    elif config.model.model_name_or_path == "microsoft/wavlm-base-plus":
        checkpoint_path = os.path.join(base_dir, "checkpoints", "SSL4PR-wavlm-base", f"fold_{fold_num}.pt")
    else:
        raise ValueError(f"Unknown model: {config.model.model_name_or_path}")

    val_path = os.path.join(base_dir, "pcgita_splits", f"TRAIN_TEST_{fold_num}", "test.csv")
    return checkpoint_path, val_path


if __name__ == "__main__":

    # Load config and set seed
    config = add_arguments()
    config = Dict(config)
    set_all_seeds(config.training.seed)

    # Define class mapping and check if it's a binary classification task
    if config.training.label_key == "status":
        class_mapping = {'hc': 0, 'pd': 1}
        is_binary_classification = True
        config.model.num_classes = 2
    elif config.training.label_key == "UPDRS-speech":
        class_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        is_binary_classification = False
        config.model.num_classes = 4

    # Loop over all 10 folds
    all_metrics = []
    for fold_num in range(1, 11):
        print(f"Evaluating Fold {fold_num}...")

        # Get the correct paths for the current fold
        checkpoint_path, val_path = get_checkpoint_and_val_paths(config, fold_num)

        # Update config with the current fold's paths
        config.training.checkpoint_path = checkpoint_path
        config.data.val_path = val_path

        # Load validation dataloader
        val_dl = get_validation_dataloader(config.data.val_path, class_mapping, config)

        # Load model and move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model_checkpoint(config.training.checkpoint_path, config, device)

        # Perform inference
        reference, predictions = eval_one_epoch(model, val_dl, device, is_binary_classification)

        # Compute and print metrics for this fold
        metrics = compute_metrics(reference, predictions, is_binary_classification)
        all_metrics.append(metrics)
        print(f"Fold {fold_num} Evaluation Metrics:", metrics)

        # Free up GPU memory
        del model, val_dl, reference, predictions
        torch.cuda.empty_cache()

    # Check the structure of all_metrics
    print("All Metrics:", all_metrics)

    # Ensure all metrics have the same keys
    metrics_keys = all_metrics[0].keys()
    metrics_values = []

    for m in all_metrics:
        # Extract values and ensure they are of compatible shape
        values = [float(m[key]) if isinstance(m[key], (int, float)) else np.nan for key in metrics_keys]
        metrics_values.append(values)

    # Convert metrics to a structured array
    metrics_values = np.array(metrics_values, dtype=object)  # Use dtype=object for inhomogeneous shapes

    # Print shapes of individual metrics for debugging
    print("Shapes of individual metrics:")
    for idx, key in enumerate(metrics_keys):
        print(f"{key}: {[len(m) if isinstance(m, (list, np.ndarray)) else 1 for m in metrics_values[:, idx]]}")

    # Compute average and standard deviation for each metric
    avg_metrics = {key: np.nanmean(metrics_values[:, idx].astype(float)) for idx, key in enumerate(metrics_keys)}
    std_metrics = {key: np.nanstd(metrics_values[:, idx].astype(float)) for idx, key in enumerate(metrics_keys)}

    print("Average Metrics across all folds:", avg_metrics)
    print("Standard Deviation of Metrics across all folds:", std_metrics)

    # Free GPU memory one last time after the final computation
    torch.cuda.empty_cache()