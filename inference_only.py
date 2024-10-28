import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from addict import Dict
from captum.attr import Saliency
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm
from yaml_config_override import add_arguments

from datasets.audio_classification_dataset import AudioClassificationDataset
from models.ssl_classification_model import InvertibleTF, SSLClassificationModel


def compute_metrics(
    reference, predictions, verbose=False, is_binary_classification=False
):
    accuracy = accuracy_score(reference, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        reference, predictions, average="macro"
    )

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


def eval_one_epoch(
    model, eval_dataloader, device, loss_fn, is_binary_classification=False
):
    model.eval()

    p_bar = tqdm(eval_dataloader, total=len(eval_dataloader), ncols=100)
    eval_loss = 0.0
    reference = []
    predictions = []
    speakers = []

    tf = InvertibleTF()
    saliency = Saliency(model)

    with torch.no_grad():
        # with torch.enable_grad():
        for batch in p_bar:
            # Move tensors to the appropriate device
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            labels = batch["labels"]
            batch["input_values"], phase = tf(batch["input_values"])

            outputs = model(batch["input_values"], phase=phase)
            with torch.enable_grad():
                attr = saliency.attribute(
                    batch["input_values"],
                    # target=labels.to(torch.long),
                    additional_forward_args=(phase),
                )

            print(attr.shape)
            # breakpoint()
            n_classes = outputs.shape[-1]

            # Calculate loss
            if is_binary_classification:
                loss = loss_fn(outputs.squeeze(-1), labels)
            else:
                loss = loss_fn(outputs.view(-1, n_classes), labels.view(-1))

            eval_loss += loss.item()
            reference.extend(labels.cpu().numpy())

            # Store predictions
            if is_binary_classification:
                predictions.extend((outputs > 0.5).cpu().numpy().astype(int))
            else:
                predictions.extend(
                    torch.argmax(outputs, dim=-1).cpu().numpy().astype(int)
                )

            p_bar.set_postfix({"loss": loss.item()})

    return eval_loss / len(eval_dataloader), reference, predictions, speakers


def get_dataloaders(test_path, class_mapping, config):
    df_test = pd.read_csv(test_path)
    print(len(df_test))
    df_test = fix_updrs_speech_labels(df_test)
    test_paths = df_test.audio_path.values.tolist()
    test_labels = df_test[config.training.label_key].values.tolist()

    test_ds = AudioClassificationDataset(
        audio_paths=test_paths,
        labels=test_labels,
        feature_extractor_name_or_path=config.model.model_name_or_path,
        class_mapping=class_mapping,
        data_config=config.data,
        is_test=True,
    )

    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    return test_dl


def fix_updrs_speech_labels(df, remove_types=["words"]):
    for r_type in remove_types:
        df = df[~df["audio_path"].str.contains(r_type)]
    df["UPDRS-speech"] = df["UPDRS-speech"].fillna(0).astype(int)
    return df


def get_speaker_disease_grade(test_path):
    df_test = pd.read_csv(test_path)
    df_test["UPDRS-speech"].replace(np.nan, -1, inplace=True)
    return dict(zip(df_test.speaker_id, df_test["UPDRS-speech"]))


def main(config):
    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.training.use_cuda else "cpu"
    )
    print(f"Using device: {device}")

    # Create class mapping
    if config.training.label_key == "status":
        class_mapping = {"hc": 0, "pd": 1}
        is_binary_classification = True
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif config.training.label_key == "UPDRS-speech":
        class_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
        is_binary_classification = False
        loss_fn = torch.nn.CrossEntropyLoss()

    overall_metrics = {
        metric: []
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "sensitivity",
            "specificity",
        ]
    }

    for test_fold in range(1, 11):
        # Load fold model
        config.model.num_classes = len(class_mapping)

        model = SSLClassificationModel(config=config)
        fold_path = f"{config.training.checkpoint_path}fold_{test_fold}.pt"
        model.load_state_dict(
            torch.load(fold_path, map_location="cpu", weights_only=False)
        )
        model.to(device)

        # Create dataloader
        test_path = f"{config.data.fold_root_path}/TRAIN_TEST_{test_fold}/test.csv"
        test_dl = get_dataloaders(test_path, class_mapping, config)

        # Evaluate
        test_loss, test_reference, test_predictions, test_speaker = eval_one_epoch(
            model=model,
            eval_dataloader=test_dl,
            device=device,
            loss_fn=loss_fn,
            is_binary_classification=is_binary_classification,
        )

        # Compute metrics
        metrics = compute_metrics(
            test_reference,
            test_predictions,
            verbose=False,
            is_binary_classification=is_binary_classification,
        )

        # Print metrics
        print(f"-" * 20, f"Fold {test_fold} Results", "-" * 20)
        for k, v in metrics.items():
            print(f"{k}: {v}")
            overall_metrics[k].append(v)

        # clear memory
        del model
        torch.cuda.empty_cache()

    # Print overall metrics
    print("\nOverall Metrics:")
    for k, v in overall_metrics.items():
        print(f"{k}: {np.mean(v)*100:.3f}")


if __name__ == "__main__":
    config = add_arguments()
    config = Dict(config)
    main(config)
