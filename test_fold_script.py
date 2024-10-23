import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from datasets.audio_classification_dataset import AudioClassificationDataset
from models.ssl_classification_model import SSLClassificationModel
import pandas as pd
from yaml_config_override import add_arguments
from addict import Dict
import numpy as np 
import os
import random
from sklearn.model_selection import train_test_split


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

# def eval_one_epoch(model, eval_dataloader, device, is_binary_classification=False):
#     model.eval()
#     eval_loss = 0.0
#     reference = []
#     predictions = []
#     with torch.no_grad():
#         for batch in tqdm(eval_dataloader, total=len(eval_dataloader), ncols=100):
#             batch = {k: v.to(device) for k, v in batch.items()}
#             labels = batch["labels"]
#             outputs = model(batch)
#             n_classes = outputs.shape[-1]
#             if is_binary_classification:
#                 predictions.extend((outputs > 0.5).cpu().numpy().astype(int))
#             else:
#                 predictions.extend(torch.argmax(outputs, dim=-1).cpu().numpy().astype(int))
#             reference.extend(labels.cpu().numpy())
#     return reference, predictions

def eval_one_epoch(model, eval_dataloader, device, loss_fn, experiment=None, is_binary_classification=False):
    model.eval()

    p_bar = tqdm(eval_dataloader, total=len(eval_dataloader), ncols=100)
    eval_loss = 0.0
    reference = []
    predictions = []

    with torch.no_grad():
        for batch in p_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            
            outputs = model(batch)
            n_classes = outputs.shape[-1]

            if is_binary_classification: loss = loss_fn(outputs.squeeze(-1), labels)
            else: loss = loss_fn(outputs.view(-1, n_classes), labels.view(-1))

            eval_loss += loss.item()
            reference.extend(labels.cpu().numpy())
            if is_binary_classification: predictions.extend( (outputs > 0.5).cpu().numpy().astype(int) )
            else: predictions.extend(torch.argmax(outputs, dim=-1).cpu().numpy().astype(int))

            p_bar.set_postfix({"loss": loss.item()})

    return eval_loss / len(eval_dataloader), reference, predictions

# def compute_metrics(reference, predictions, is_binary_classification=False):
#     accuracy = accuracy_score(reference, predictions)
#     precision, recall, f1, _ = precision_recall_fscore_support(reference, predictions, average="macro")
#     cm = confusion_matrix(reference, predictions)
#     metrics = {
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "confusion_matrix": cm,
#     }
#     return metrics

def compute_metrics(reference, predictions, verbose=False, is_binary_classification=False):
    
    accuracy = accuracy_score(reference, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(reference, predictions, average="macro")
    
    if is_binary_classification:
        roc_auc = roc_auc_score(reference, predictions)
        cm = confusion_matrix(reference, predictions)
        tp = cm[1, 1]
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
    else:
        print("ROC AUC is not defined for multiclass classification")
        roc_auc = 0.0
        sensitivity = 0.0
        specificity = 0.0
        
    if verbose:
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        print(f"ROC AUC: {roc_auc}")
        print(f"Sensitivity: {sensitivity}")
        print(f"Specificity: {specificity}")
        
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }

# def get_validation_dataloader(val_path, class_mapping, config):
#     df_val = pd.read_csv(val_path)
#     val_paths, val_labels = df_val.audio_path.values.tolist(), df_val[config.training.label_key].values.tolist()
#     val_ds = AudioClassificationDataset(
#         audio_paths=val_paths,
#         labels=val_labels,
#         feature_extractor_name_or_path=config.model.model_name_or_path,
#         class_mapping=class_mapping,
#         data_config=config.data,
#         is_test=True
#     )
#     val_dl = torch.utils.data.DataLoader(
#         val_ds,
#         batch_size=config.training.batch_size,
#         shuffle=False,
#         num_workers=config.training.num_workers,
#         pin_memory=config.training.pin_memory,
#     )
#     return val_dl


def fix_updrs_speech_labels(df):
    df["UPDRS-speech"] = df["UPDRS-speech"].fillna(0)
    df["UPDRS-speech"] = df["UPDRS-speech"].astype(int)
    return df

def get_dataloaders(train_path, test_path, class_mapping, config):
        
    df_test = pd.read_csv(test_path)
    # remove the rows containing "words" in the audio_path
    df_test = fix_updrs_speech_labels(df_test)
    test_paths = df_test.audio_path.values.tolist()
    test_labels = df_test[config.training.label_key].values.tolist()
    
    df_train = pd.read_csv(train_path)
    df_train = fix_updrs_speech_labels(df_train)
    if config.training.validation.active:
        if config.training.validation.validation_type == "speaker":
            paths, speaker_ids = df_train.audio_path.values.tolist(), df_train.speaker_id.values.tolist()
            labels = df_train[config.training.label_key].values.tolist()
            unique_speaker_ids = list(set(speaker_ids))
            unique_speaker_ids.sort()
            random.shuffle(unique_speaker_ids)
            train_speaker_ids, val_speaker_ids = train_test_split(
                unique_speaker_ids, test_size=config.training.validation.validation_split, random_state=42
            )
            t_paths, t_labels, v_paths, v_labels = [], [], [], []
            for path, label, speaker_id in zip(paths, labels, speaker_ids):
                if speaker_id in train_speaker_ids:
                    t_paths.append(path)
                    t_labels.append(label)
                else:
                    v_paths.append(path)
                    v_labels.append(label)
        elif config.training.validation.validation_type == "random":
            # just 90/10 split - on paths directly
            paths, labels = df_train.audio_path.values.tolist(), df_train[config.training.label_key].values.tolist()
            t_paths, v_paths, t_labels, v_labels = train_test_split(
                paths, labels, test_size=config.training.validation.validation_split, random_state=42
            )
        else:
            raise ValueError(f"Validation is active but validation type: {config.training.validation.validation_type} is not supported")
    else:
        t_paths, t_labels = df_train.audio_path.values.tolist(), df_train[config.training.label_key].values.tolist()
        v_paths, v_labels = [], []
        
    # set model.num_classes according to the number of classes in the dataset
    config.model.num_classes = len(set(t_labels))
    
    # create datasets
    t_ds = AudioClassificationDataset(
        audio_paths=t_paths,
        labels=t_labels,
        feature_extractor_name_or_path=config.model.model_name_or_path,
        class_mapping=class_mapping,
        data_config=config.data,
    )
    
    if config.training.validation.active:
        v_ds = AudioClassificationDataset(
            audio_paths=v_paths,
            labels=v_labels,
            feature_extractor_name_or_path=config.model.model_name_or_path,
            class_mapping=class_mapping,
            data_config=config.data,
            is_test=True
        )
    
    test_ds = AudioClassificationDataset(
        audio_paths=test_paths,
        labels=test_labels,
        feature_extractor_name_or_path=config.model.model_name_or_path,
        class_mapping=class_mapping,
        data_config=config.data,
        is_test=True,
    )
    
    # create dataloaders
    train_dl = torch.utils.data.DataLoader(
        t_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    if config.training.validation.active:
        val_dl = torch.utils.data.DataLoader(
            v_ds,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
        )
    else:
        val_dl = None
    
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )
    
    return train_dl, val_dl, test_dl


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
    for fold_num in range(1, config.data.num_folds+1):
        
        # info about the fold
        fold_path = config.data.fold_root_path + f"/TRAIN_TEST_{fold_num}/"
        train_path = fold_path + "train.csv"
        test_path = fold_path + "test.csv"
        train_dl, val_dl, test_dl = get_dataloaders(train_path, test_path, class_mapping, config)
        val_dl = test_dl

        # Get the correct paths for the current fold
        checkpoint_path, val_path = get_checkpoint_and_val_paths(config, fold_num)

        # Update config with the current fold's paths
        config.training.checkpoint_path = checkpoint_path
        config.data.val_path = val_path

        # # Load validation dataloader
        # val_dl = get_validation_dataloader(config.data.val_path, class_mapping, config)

        # Load model and move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model_checkpoint(config.training.checkpoint_path, config, device)

        # Create loss function
        # create loss function
        if is_binary_classification: loss_fn = torch.nn.BCELoss()
        else: loss_fn = torch.nn.CrossEntropyLoss()

        # Perform inference
        val_loss, reference, predictions = eval_one_epoch(model, val_dl, device, loss_fn=loss_fn, is_binary_classification=is_binary_classification)

        # Compute and print metrics for this fold
        metrics = compute_metrics(reference, predictions, verbose=True,  is_binary_classification=is_binary_classification)
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

    accuracies =  [m["accuracy"] for m in all_metrics]
    avg_accuracy = np.mean(accuracies)
    print(f"Average Accuracy across all folds: {avg_accuracy*100:.2f}")
    # Compute average and standard deviation for each metric
    avg_metrics = {key: np.nanmean(metrics_values[:, idx].astype(float)) for idx, key in enumerate(metrics_keys)}
    std_metrics = {key: np.nanstd(metrics_values[:, idx].astype(float)) for idx, key in enumerate(metrics_keys)}


    print("Average Metrics across all folds:", avg_metrics)
    print("Standard Deviation of Metrics across all folds:", std_metrics)

    # Save results in a .txt file as a table
    with open("metrics_summary.txt", "w") as file:
        file.write(f"{'Metric':<20}{'Average':<20}{'Std Dev':<20}\n")
        file.write("="*60 + "\n")
        for key in metrics_keys:
            avg = avg_metrics[key]
            std = std_metrics[key]
            file.write(f"{key:<20}{avg:<20.4f}{std:<20.4f}\n")

    print("Metrics saved to metrics_summary.txt")

    # Free GPU memory one last time after the final computation
    torch.cuda.empty_cache()

    # Free GPU memory one last time after the final computation
    torch.cuda.empty_cache()