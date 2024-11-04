import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from addict import Dict
from captum.attr import IntegratedGradients, Saliency, NoiseTunnel, GuidedBackprop
from captum.attr import visualization as viz
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from speechbrain.utils.metric_stats import MetricStats
from tqdm import tqdm
from yaml_config_override import add_arguments

from datasets.audio_classification_dataset import AudioClassificationDataset
from models.ssl_classification_model import (InvertibleTF,
                                             SSLClassificationModel)

import librosa
import librosa.display

import torchaudio

from speechbrain.processing.features import ISTFT, STFT
import speechbrain as sb

from utils import viz

import quantus 


eps = 1e-10

int_strategies = {"saliency": Saliency, "ig": IntegratedGradients, "gbp": GuidedBackprop}
adds_params = {"saliency": {}, "ig": {"n_steps": 5}, "gbp": {}} 

strategy = "gbp"


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


def compute_fidelity(theta_out, predictions):
    """Computes top-`k` fidelity of interpreter."""
    pred_cl = (predictions > 0.5).float()
    k_top = (theta_out > 0.5).float()

    # 1 element for each sample in batch, is 0 if pred_cl is in top k
    temp = (k_top - pred_cl.unsqueeze(1) == 0).sum(1)

    return temp


@torch.no_grad()
def compute_faithfulness(predictions, predictions_masked):
    "This function implements the faithful metric (FF) used in the L-MAC paper."
    # get the prediction indices
    pred_cl = (predictions > 0.5).float()
    predictions_masked_selected = (predictions_masked > 0.5).float()

    faithfulness = (pred_cl - predictions_masked_selected)#.squeeze(dim=1)

    return faithfulness


@torch.no_grad()
def compute_AD(theta_out, predictions):
    """Computes top-`k` fidelity of interpreter."""
    pred_cl = (predictions > 0.5).float()
    theta_out = (theta_out > 0.5).float()

    # 1 element for each sample in batch, is 0 if pred_cl is in top k
    temp = (F.relu(pred_cl - theta_out) / (pred_cl + eps)) * 100

    return temp


@torch.no_grad()
def compute_AI(theta_out, predictions):
    """Computes top-`k` fidelity of interpreter."""
    pc = (predictions > 0.5).float()
    oc = (theta_out > 0.5).float()

    # 1 element for each sample in batch, is 0 if pred_cl is in top k
    temp = (pc < oc).float() * 100

    return temp


@torch.no_grad()
def compute_AG(theta_out, predictions):
    """Computes top-`k` fidelity of interpreter."""
    pc = (predictions > 0.5).float()
    oc = (theta_out > 0.5).float()

    # 1 element for each sample in batch, is 0 if pred_cl is in top k
    temp = (F.relu(oc - pc) / (1 - pc + eps)) * 100

    return temp


@torch.no_grad()
def compute_sparseness(attr, X, y):
    """Computes the SPS metric used in the L-MAC paper."""
    sparseness = quantus.Sparseness(
        return_aggregate=True, abs=True
    )
    device = X.device
    attr = ( 
        attr.unsqueeze(1)
        .clone()
        .detach()
        .cpu()
        .numpy()

    )
    if attr.sum() > 0:
        X = X[:, : attr.shape[2], :]
        X = X.unsqueeze(1)
        quantus_inp = {
            "model": None,
            "x_batch": X.clone()
            .detach()
            .cpu()
            .numpy(),  # quantus expects the batch dim
            "a_batch": attr,
            "y_batch": y.numpy(),
            "softmax": False,
            "device": device,
        }
        return torch.Tensor([sparseness(**quantus_inp)[0]]).float()
    else:
        print("all zeros saliency map")
        return torch.zeros([0])

@torch.no_grad()
def compute_complexity(attr, X, y):
    """Computes the COMP metric used in L-MAC paper"""
    complexity = quantus.Complexity(
        return_aggregate=True, abs=True
    )
    device = X.device
    attr = (
        attr.unsqueeze(1)
        .clone()
        .detach()
        .cpu()
        .numpy()
    )
    if attr.sum() > 0:
        X = X[:, : attr.shape[2], :]
        X = X.unsqueeze(1)
        quantus_inp = {
            "model": None,
            "x_batch": X.clone()
            .detach()
            .cpu()
            .numpy(),  # quantus expects the batch dim
            "a_batch": attr,
            "y_batch": y.numpy(),
            "softmax": False,
            "device": device,
        }

        return torch.Tensor([complexity(**quantus_inp)[0]]).float()
    else:
        print("all zeros saliency map")
        return torch.zeros([0])


@torch.no_grad()
def accuracy_value(predict, target):
    """Computes Accuracy"""
    predict = predict.argmax(1)

    return (predict.unsqueeze(1) == target).float().squeeze(1)


def compute_interpretability_metrics(
    predictions, predictions_masked, theta_out, reference, attributions, originals
):
    """
    Compute interpretability metrics.
    """

    return {
        "AD": compute_AD(theta_out, predictions).mean().item(),
        "AI": compute_AI(theta_out, predictions).mean().item(),
        "AG": compute_AG(theta_out, predictions).mean().item(),
        "inp_fid": compute_fidelity(theta_out, predictions).float().mean().item(),
        "faithfulness": compute_faithfulness(predictions, predictions_masked)
        .mean()
        .item(),
        "sparseness": compute_sparseness(attributions, originals, reference).mean().item(),
        "complexity": compute_complexity(attributions, originals, reference).mean().item()

    }


def eval_one_epoch(
    model, eval_dataloader, device, loss_fn, fold_dir, is_binary_classification=False
):
    model.eval()
    if strategy == "ig":
        model = model.double()

    p_bar = tqdm(eval_dataloader, total=len(eval_dataloader), ncols=100)
    eval_loss = 0.0
    reference = []
    predictions = []
    speakers = []
    predictions_masked_list = []
    theta_list = []
    outputs_list = []
    attributions = []
    originals = []


    tf = InvertibleTF()
    saliency = int_strategies[strategy](model)

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
            if strategy == "ig":
                batch["input_values"] = batch["input_values"].double()
                phase = phase.double()

            outputs = model(batch["input_values"], phase=phase)
            with torch.enable_grad():
                attr = saliency.attribute(
                    batch["input_values"],
                    # target=labels.to(torch.long),
                    additional_forward_args=(phase),
                    **adds_params[strategy],
                )

            attributions.extend(attr.cpu())
            originals.extend(batch["input_values"].cpu())

            predictions_masked = model(
                batch["input_values"], mask=1 - attr, phase=phase
            )  # mask_out
            theta = model(batch["input_values"], mask=attr, phase=phase)  # mask_in

            predictions_masked_list.extend(predictions_masked.cpu())
            theta_list.extend(theta.cpu())
            outputs_list.extend(outputs.cpu())

            print(attr.shape)

            # original and attributions
            original = batch["input_values"].squeeze().cpu().numpy()
            attr = attr.squeeze().cpu().numpy()

    
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

            
            viz.save_interpretations_for_conditions(fold_dir, original, attr, phase, labels, predictions, tf, sample_rate=16000)

            p_bar.set_postfix({"loss": loss.item()})

    # predictions_masked = np.array(predictions_masked_list.cpu())
    # theta = np.array(theta_list.cpu())

    predictions_masked_tensor = torch.stack(predictions_masked_list)
    theta_tensor = torch.stack(theta_list)
    outputs_tensor = torch.stack(outputs_list)
    attributions_tensor = torch.stack(attributions)
    originals_tensor = torch.stack(originals)

    return (
        eval_loss / len(eval_dataloader),
        reference,
        predictions,
        speakers,
        #attr,
        #original,
        predictions_masked_tensor,
        theta_tensor,
        outputs_tensor,
        attributions_tensor,
        originals_tensor
    )


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

    # Pretrained model
    pretrained_model = config.model.model_name_or_path

    if "wavlm" in pretrained_model:
        pretrained_model = "wavlm"
    elif "hubert" in pretrained_model:
        pretrained_model = "hubert"

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
            "AI",
            "AD",
            "AG",
            "inp_fid",
            "faithfulness",
            "sparseness",
            "complexity"
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


        # # Create fold_dir if it does not exist
        # fold_dir = os.path.join(int_dir, f"fold_{test_fold}")
        # if not os.path.exists(fold_dir):
        #     os.makedirs(fold_dir)
    
        # Create interptretations dir 
        # Define the base directory for interpretations
        base_dir = "results"

        # Ensure the base directory exists
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        model_dir = os.path.join(base_dir, pretrained_model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        int_dir = os.path.join(model_dir, strategy)
        if not os.path.exists(int_dir):
            os.makedirs(int_dir)
            
        # Save attributions 
        attributions_dir = os.path.join(int_dir, "attributions")
        if not os.path.exists(attributions_dir):
            os.makedirs(attributions_dir)
    
        attributions_fold_dir = os.path.join(attributions_dir, f"fold_{test_fold}")
        if not os.path.exists(attributions_fold_dir):
            os.makedirs(attributions_fold_dir) 

        visualizations_dir = os.path.join(int_dir, "interpretations")
        if not os.path.exists(visualizations_dir):
            os.makedirs(visualizations_dir)
        
       
        visualizations_fold_dir = os.path.join(visualizations_dir, f"fold_{test_fold}")
        if not os.path.exists(visualizations_fold_dir):
            os.makedirs(visualizations_fold_dir)

        # Evaluate
        (
            test_loss,
            test_reference,
            test_predictions,
            test_speaker,
            #attr,
            #original,
            predictions_masked,
            theta,
            outputs, 
            attributions,
            originals
        ) = eval_one_epoch(
            model=model,
            eval_dataloader=test_dl,
            device=device,
            loss_fn=loss_fn,
            fold_dir = visualizations_fold_dir,
            is_binary_classification=is_binary_classification,
        )

        
        # Save attributions
        torch.save(attributions, os.path.join(attributions_fold_dir,  "attributions.pt"))

        # Save gold labels
        torch.save(test_reference, os.path.join(attributions_fold_dir, "gold_labels.pt"))

        # Save originals 
        torch.save(originals, os.path.join(attributions_fold_dir, "originals.pt"))

        # Compute metrics
        metrics = compute_metrics(
            test_reference,
            test_predictions,
            verbose=False,
            is_binary_classification=is_binary_classification,
        )

        test_reference = torch.tensor(np.stack(test_reference))
        # Compure interpretability metrics
        interpretability_metrics = compute_interpretability_metrics(
            predictions=outputs,
            predictions_masked=predictions_masked,
            theta_out=theta,
            reference=test_reference,
            attributions=attributions, 
            originals=originals
        )

        # Print metrics
        print(f"-" * 20, f"Fold {test_fold} Results", "-" * 20)
        for k, v in metrics.items():
            print(f"{k}: {v}")
            overall_metrics[k].append(v)

        # Print interpretability metrics
        print(f"-" * 20, f"Fold {test_fold} Interpretability Metrics", "-" * 20)
        for k, v in interpretability_metrics.items():
            print(f"{k}: {v}")
            overall_metrics[k].append(v)

        # clear memory
        del model
        torch.cuda.empty_cache()

    # Print overall metrics
    print("\nOverall Metrics:")
    for k, v in overall_metrics.items():
        if k in {"AD", "AI", "AG", "inp_fid", "faithfulness", "sparseness", "complexity"}:
            mean_value = np.mean(v)
            std_value = np.std(v)
            print(f"{k}: Mean = {mean_value:.3f}, Std = {std_value:.3f}")
        else:
            mean_value = np.mean(v) * 100
            std_value = np.std(v) * 100
            print(f"{k}: Mean = {mean_value:.3f}%, Std = {std_value:.3f}%")


if __name__ == "__main__":
    config = add_arguments()
    config = Dict(config)
    main(config)
