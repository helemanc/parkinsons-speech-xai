import json
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantus
import speechbrain as sb
import torch
import torch.nn.functional as F
import torchaudio
from addict import Dict
from captum.attr import (
    GradientShap,
    GuidedBackprop,
    GuidedGradCam,
    IntegratedGradients,
    NoiseTunnel,
    Saliency,
)
from captum.attr import visualization as viz
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from speechbrain.processing.features import ISTFT, STFT
from speechbrain.utils.metric_stats import MetricStats
from tqdm import tqdm
from yaml_config_override import add_arguments

from datasets.audio_classification_dataset import AudioClassificationDataset
from models.ssl_classification_model import InvertibleTF, SSLClassificationModel
from utils import viz

eps = 1e-10

int_strategies = {
    "saliency": Saliency,
    "gbp": GuidedBackprop,
    # "ig": IntegratedGradients,
    # "shap": GradientShap,
    # "smoothgrad": NoiseTunnel,
    # "ggc": GuidedGradCam,
}
adds_params = {
    "saliency": {},
    "gbp": {},
    # "ig": {"n_steps": 5},
    # "shap": {},
    # "smoothgrad": {"nt_type": "smoothgrad", "nt_samples": 10},
    # "ggc": {},
}

overlap = True


def compute_overlap_attribution(attribution1, attribution2, overlap_strategy = "sum"):
    # Ensure both attributions are of the same shape
    assert (
        attribution1.shape == attribution2.shape
    ), "Attributions must have the same shape"

    # Calculate the overlap attribution
    # iou = (attribution1 * attribution2) / (
    #     attribution1 + attribution2 + 1e-10
    # ) #IoU
    a1 = (attribution1 >= 0.4 * attribution1.reshape((attribution1.shape[0], -1)).max(-1).values[..., None, None]).float()
    a2 = (attribution2 >= 0.4 * attribution1.reshape((attribution2.shape[0], -1)).max(-1).values[..., None, None]).float()

    iou = (a1 * a2).sum() / ((a1 + a2) > 0).float().sum()


    if overlap_strategy == "sum":
        overlap_attribution = attribution1 + attribution2
    elif overlap_strategy == "prod":
        overlap_attribution = attribution1 * attribution2
    
    overlap_attribution = overlap_attribution/overlap_attribution.max()



    return iou, overlap_attribution


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
    temp = ((k_top - pred_cl.unsqueeze(1)) == 0).sum(1)

    return temp


@torch.no_grad()
def compute_faithfulness(predictions, predictions_masked):
    "This function implements the faithful metric (FF) used in the L-MAC paper."
    # get the prediction indices
    # pred_cl = (predictions > 0.5).float()
    # predictions_masked_selected = (predictions_masked > 0.5).float()
    ones = (predictions > 0.5).float()
    faithfulness = (predictions - predictions_masked) * ones  # .squeeze(dim=1)

    zeros = (predictions <= 0.5).float()
    # 1 - px - (1-pxo) = pxo - px
    faithfulness +=  (predictions_masked - predictions) * zeros

    return faithfulness


@torch.no_grad()
def compute_AD(theta_out, predictions):
    """Computes top-`k` fidelity of interpreter."""
    pred_cl = predictions
    theta_out = theta_out

    # 1 element for each sample in batch, is 0 if pred_cl is in top k
    temp = (F.relu(pred_cl - theta_out) / (pred_cl + eps)) * 100

    return temp


@torch.no_grad()
def compute_AI(theta_out, predictions):
    """Computes top-`k` fidelity of interpreter."""
    pc = predictions
    oc = theta_out

    # 1 element for each sample in batch, is 0 if pred_cl is in top k
    temp = (pc < oc).float() * 100

    return temp


@torch.no_grad()
def compute_AG(theta_out, predictions):
    """Computes top-`k` fidelity of interpreter."""
    pc = predictions
    oc = theta_out

    # 1 element for each sample in batch, is 0 if pred_cl is in top k
    temp = (F.relu(oc - pc) / (1 - pc + eps)) * 100

    return temp


@torch.no_grad()
def compute_sparseness(attr, X, y):
    """Computes the SPS metric used in the L-MAC paper."""
    sparseness = quantus.Sparseness(return_aggregate=True, abs=True)
    device = X.device
    attr = attr.unsqueeze(1).clone().detach().cpu().numpy()
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
    complexity = quantus.Complexity(return_aggregate=True, abs=True)
    device = X.device
    attr = attr.unsqueeze(1).clone().detach().cpu().numpy()
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
        "sparseness": compute_sparseness(attributions, originals, reference)
        .mean()
        .item(),
        "complexity": compute_complexity(attributions, originals, reference)
        .mean()
        .item(),
    }


def eval_one_epoch_combined(
    model,
    dataloader,
    device,
    loss_fn,
    fold_dir,
    strategy=None,
    overlap_name=None,
    is_binary_classification=False,
):
    model.eval()

    if (strategy == "ig") or (overlap_name and "ig" in overlap_name):
        model = model.double()

    p_bar = tqdm(dataloader, total=len(dataloader), ncols=100)
    eval_loss = 0.0
    reference, predictions, speakers = [], [], []
    predictions_masked_list, theta_list, outputs_list = [], [], []
    attributions, originals, phases = [], [], []

    tf = InvertibleTF()

    # Set strategy name and initialize saliency if using `eval` strategy
    if strategy:
        if strategy == "smoothgrad":
            saliency = int_strategies[strategy](Saliency(model))
        elif strategy == "ggc":
            if "hubert" in str(model.ssl_model.__class__):
                saliency = int_strategies[strategy](
                    model, model.ssl_model.encoder.layers[-1]
                )
            else:
                raise NotImplementedError(
                    "GradCAM not implemented for models other than HuBert."
                )
        else:
            saliency = int_strategies[strategy](model)

        eval_mode = True
    else:
        strategy = overlap_name
        eval_mode = False

    with torch.no_grad():
        for batch in p_bar:
            # Process the batch according to overlap/eval mode
            if eval_mode:
                # Standard evaluation mode
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                labels = batch["labels"]
                batch["input_values"], phase = tf(batch["input_values"])

            else:
                # Overlap mode
                overlap_attributions, input_values, labels, phases = batch
                batch_dict = {
                    "overlap_attributions": overlap_attributions,
                    "input_values": input_values,
                    "labels": labels,
                    "phases": phases,
                }
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch_dict.items()
                }
                labels = batch["labels"]
                phase = batch["phases"]
                attr = batch["overlap_attributions"]

            # Double tensor for "ig" strategy
            if "ig" in strategy:
                batch["input_values"] = batch["input_values"].double()
                phase = phase.double()

            # Forward pass
            outputs = model(batch["input_values"], phase=phase)

            # Attribution handling
            if eval_mode:
                if strategy == "shap":
                    adds_params["shap"]["baselines"] = torch.randn_like(
                        batch["input_values"]
                    )

                with torch.enable_grad():
                    attr = saliency.attribute(
                        batch["input_values"],
                        additional_forward_args=(phase),
                        **adds_params[strategy],
                    )

                attr = attr.abs().float()
                attr = attr / attr.max()

            attributions.extend(attr.cpu())
            originals.extend(batch["input_values"].cpu())

            if eval_mode:
                phases.extend(phase.cpu())

            # Masked predictions
            predictions_masked = model(
                batch["input_values"],
                mask=1 - attr,
                phase=phase,
                # batch["input_values"], mask=1.0 - attr, phase=phase
            )  # mask_out
            theta = model(batch["input_values"], mask=attr, phase=phase)  # mask_in

            predictions_masked_list.extend(predictions_masked.cpu())
            theta_list.extend(theta.cpu())
            outputs_list.extend(outputs.cpu())

            # Loss calculation
            n_classes = outputs.shape[-1]
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

            # Visualization
            original = batch["input_values"].squeeze().cpu().numpy()
            attr = attr.squeeze().cpu().numpy()
            viz.save_interpretations_for_conditions(
                fold_dir,
                original,
                attr,
                phase,
                labels,
                predictions,
                tf,
                sample_rate=16000,
            )

            p_bar.set_postfix({"loss": loss.item()})

    predictions_masked_tensor = torch.stack(predictions_masked_list)
    theta_tensor = torch.stack(theta_list)
    outputs_tensor = torch.stack(outputs_list)
    attributions_tensor = torch.stack(attributions)
    originals_tensor = torch.stack(originals)

    return (
        eval_loss / len(dataloader),
        reference,
        predictions,
        speakers,
        predictions_masked_tensor,
        theta_tensor,
        outputs_tensor,
        attributions_tensor,
        originals_tensor,
        torch.stack(phases) if eval_mode else None,
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


# Helper Functions
def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load_model(test_fold, config, device, class_mapping):
    """Load the model for a specific fold."""
    config.model.num_classes = len(class_mapping)
    model = SSLClassificationModel(config=config)
    fold_path = f"{config.training.checkpoint_path}fold_{test_fold}.pt"
    model.load_state_dict(
        torch.load(fold_path, map_location=device, weights_only=False)
    )
    return model.to(device)


def save_tensor(tensor, path):
    """Save a tensor to a file."""
    torch.save(tensor, path)


def save_metrics(overall_metrics, output_folder):
    """Save overall and average metrics to JSON files."""
    with open(f"{output_folder}/overall_metrics.json", "w") as f:
        json.dump(overall_metrics, f, indent=4)

    # Calculate average metrics
    average_metrics = {
        k: {
            "mean": np.mean(v)
            * (
                100
                if k
                not in {
                    "AD",
                    "AI",
                    "AG",
                    "inp_fid",
                    "faithfulness",
                    "sparseness",
                    "complexity",
                    "iou"
                }
                else 1
            ),
            "std": np.std(v)
            * (
                100
                if k
                not in {
                    "AD",
                    "AI",
                    "AG",
                    "inp_fid",
                    "faithfulness",
                    "sparseness",
                    "complexity",
                    "iou"
                }
                else 1
            ),
        }
        for k, v in overall_metrics.items()
    }

    with open(f"{output_folder}/average_metrics.json", "w") as f:
        json.dump(average_metrics, f, indent=4)


def compute_attributions(
    pretrained_model,
    strategy,
    config,
    device,
    loss_fn,
    class_mapping,
    overall_metrics,
    is_binary_classification,
):
    """Compute attributions and save metrics for a single fold."""

    for test_fold in range(1, 11):
        model = load_model(test_fold, config, device, class_mapping)
        test_path = f"{config.data.fold_root_path}/TRAIN_TEST_{test_fold}/test.csv"
        test_dl = get_dataloaders(test_path, class_mapping, config)

        # Create directories for results and interpretations
        base_dir = create_directory("results")
        model_dir = create_directory(os.path.join(base_dir, pretrained_model))
        int_dir = create_directory(os.path.join(model_dir, strategy))
        attributions_fold_dir = create_directory(
            os.path.join(int_dir, "attributions", f"fold_{test_fold}")
        )
        visualizations_fold_dir = create_directory(
            os.path.join(int_dir, "interpretations", f"fold_{test_fold}")
        )

        # Evaluate model and compute metrics
        (
            _,
            test_reference,
            test_predictions,
            _,
            predictions_masked,
            theta,
            outputs,
            attributions,
            originals,
            phases,
        ) = eval_one_epoch_combined(
            model=model,
            dataloader=test_dl,
            device=device,
            loss_fn=loss_fn,
            fold_dir=visualizations_fold_dir,
            is_binary_classification=is_binary_classification,
            strategy=strategy,
        )

        # Save attributions and results
        save_tensor(
            attributions, os.path.join(attributions_fold_dir, "attributions.pt")
        )
        save_tensor(
            torch.stack([torch.tensor(arr) for arr in test_reference]),
            os.path.join(attributions_fold_dir, "gold_labels.pt"),
        )
        save_tensor(originals, os.path.join(attributions_fold_dir, "originals.pt"))
        save_tensor(phases, os.path.join(attributions_fold_dir, "phases.pt"))

        # Compute and accumulate metrics
        metrics = compute_metrics(
            test_reference,
            test_predictions,
            verbose=False,
            is_binary_classification=is_binary_classification,
        )
        test_reference_tensor = torch.stack(
            [torch.tensor(arr) for arr in test_reference]
        )
        interpretability_metrics = compute_interpretability_metrics(
            predictions=outputs,
            predictions_masked=predictions_masked,
            theta_out=theta,
            reference=test_reference_tensor,
            attributions=attributions,
            originals=originals,
        )

        for k, v in {**metrics, **interpretability_metrics}.items():
            overall_metrics[k].append(v)

        # Print metrics
        print(f"-" * 20, f"Fold {test_fold} Results", "-" * 20)
        for k, v in metrics.items():
            print(f"{k}: {v}")

        print(f"-" * 20, f"Fold {test_fold} Interpretability Metrics", "-" * 20)
        for k, v in interpretability_metrics.items():
            print(f"{k}: {v}")

        # Clear model from memory
        del model
        torch.cuda.empty_cache()

    # save the average metrics to a json file
    average_metrics = {}
    for k, v in overall_metrics.items():
        if k in {
            "AD",
            "AI",
            "AG",
            "inp_fid",
            "faithfulness",
            "sparseness",
            "complexity",
        }:
            mean_value = np.mean(v)
            std_value = np.std(v)
            average_metrics[k] = {"mean": mean_value, "std": std_value}
        else:
            mean_value = np.mean(v) * 100
            std_value = np.std(v) * 100
            average_metrics[k] = {"mean": mean_value, "std": std_value}

    save_metrics(overall_metrics, int_dir)


def compute_overlap_attributions(
    pretrained_model,
    config,
    device,
    loss_fn,
    class_mapping,
    int_strategies,
    overall_metrics,
    is_binary_classification,
):
    """Compute overlap attributions between strategies and save metrics."""
    base_dir = "results"
    model_dir = create_directory(os.path.join(base_dir, pretrained_model))
    overlap_dir = create_directory(os.path.join(model_dir, "overlap"))

    for i, strategy1 in enumerate(int_strategies.keys()):
        for j, strategy2 in enumerate(int_strategies.keys()):
            if i < j:
                output_folder = create_directory(
                    os.path.join(overlap_dir, f"{strategy1}_{strategy2}", config["overlap_strategy"])
                )
                for test_fold in range(1, 11):
                    model = load_model(test_fold, config, device, class_mapping)

                    # Load attributions and labels
                    folder1, folder2 = os.path.join(model_dir, strategy1), os.path.join(
                        model_dir, strategy2
                    )
                    attribution1 = torch.load(
                        os.path.join(
                            folder1,
                            "attributions",
                            f"fold_{test_fold}",
                            "attributions.pt",
                        )
                    )
                    attribution2 = torch.load(
                        os.path.join(
                            folder2,
                            "attributions",
                            f"fold_{test_fold}",
                            "attributions.pt",
                        )
                    )
                    labels1 = torch.load(
                        os.path.join(
                            folder1,
                            "attributions",
                            f"fold_{test_fold}",
                            "gold_labels.pt",
                        )
                    )
                    labels2 = torch.load(
                        os.path.join(
                            folder2,
                            "attributions",
                            f"fold_{test_fold}",
                            "gold_labels.pt",
                        )
                    )
                    originals1 = torch.load(
                        os.path.join(
                            folder1, "attributions", f"fold_{test_fold}", "originals.pt"
                        )
                    )
                    originals2 = torch.load(
                        os.path.join(
                            folder2, "attributions", f"fold_{test_fold}", "originals.pt"
                        )
                    )
                    phases1 = torch.load(
                        os.path.join(
                            folder1, "attributions", f"fold_{test_fold}", "phases.pt"
                        )
                    )
                    phases2 = torch.load(
                        os.path.join(
                            folder2, "attributions", f"fold_{test_fold}", "phases.pt"
                        )
                    )
                    assert torch.equal(labels1, labels2), "Labels must match"
                    assert torch.equal(originals1, originals2), "Originals must match"
                    assert torch.equal(phases1, phases2), "Phases must match"

                    # Compute overlap
                    iou, overlaps = compute_overlap_attribution(attribution1, attribution2, config["overlap_strategy"])
                    dataloader = torch.utils.data.DataLoader(
                        torch.utils.data.TensorDataset(
                            overlaps, originals1, labels1, phases1
                        ),
                        batch_size=config.training.batch_size,
                        shuffle=False,
                        )
                    

                    attributions_dir = create_directory(
                        os.path.join(output_folder, "attributions")
                    )
                    attributions_fold_dir = create_directory(
                        os.path.join(attributions_dir, f"fold_{test_fold}")
                    )
                    torch.save(
                        overlaps, os.path.join(attributions_fold_dir, "attributions.pt")
                    )
                    # Evaluate on overlap data
                    visualizations_fold_dir = create_directory(
                        os.path.join(
                            output_folder, "interpretations", f"fold_{test_fold}"
                        )
                    )
                    
                    
                    (
                        _,
                        test_reference,
                        test_predictions,
                        _,
                        predictions_masked,
                        theta,
                        outputs,
                        attributions,
                        originals,
                        _,
                    ) = eval_one_epoch_combined(
                        model=model,
                        overlap_name=f"{strategy1}_{strategy2}",
                        dataloader=dataloader,
                        device=device,
                        loss_fn=loss_fn,
                        fold_dir=visualizations_fold_dir,
                        is_binary_classification=is_binary_classification,
                    )

                    viz.plot_comparative_maps(originals[0], attribution1, attribution2, strategy1, strategy2, labels1[0], test_predictions[0],
                          sample_rate=16000, hop_length_samples=185, win_length_samples=371, 
                          save_path=os.path.join(visualizations_fold_dir, "comparative_maps.png"))

                    # Compute metrics
                    metrics = compute_metrics(
                        test_reference,
                        test_predictions,
                        verbose=False,
                        is_binary_classification=is_binary_classification,
                    )

                    test_reference_tensor = torch.stack(
                        [torch.tensor(arr) for arr in test_reference]
                    )
                    # Compute interpretability metrics
                    interpretability_metrics = compute_interpretability_metrics(
                        predictions=outputs,
                        predictions_masked=predictions_masked,
                        theta_out=theta,
                        reference=test_reference_tensor,
                        attributions=attributions,
                        originals=originals,
                    )

                    interpretability_metrics["iou"] = iou.float().mean().item()

                    # Print metrics
                    print(f"-" * 20, f"Fold {test_fold} Results", "-" * 20)
                    for k, v in metrics.items():
                        print(f"{k}: {v}")
                        overall_metrics[k].append(v)

                    # Print interpretability metrics
                    print(
                        f"-" * 20,
                        f"Fold {test_fold} Interpretability Metrics",
                        "-" * 20,
                    )
                    for k, v in interpretability_metrics.items():
                        print(f"{k}: {v}")
                        overall_metrics[k].append(v)

                    # Clear model from memory
                    del model
                    torch.cuda.empty_cache()

                    print(f"-" * 20, f"Fold {test_fold} Results", "-" * 20)
                    for k, v in interpretability_metrics.items():
                        print(f"{k}: {v}")

               

                # Save all metrics
                save_metrics(overall_metrics, output_folder)


# Main Processing Function


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
            "complexity",
            "iou"
        ]
    }

    if not overlap:
        compute_attributions(
            pretrained_model,
            config.training.strategy_key,
            config,
            device,
            loss_fn,
            class_mapping,
            overall_metrics,
            is_binary_classification,
        )
    else:
        compute_overlap_attributions(
            pretrained_model,
            config,
            device,
            loss_fn,
            class_mapping,
            int_strategies,
            overall_metrics,
            is_binary_classification,
        )


if __name__ == "__main__":
    config = add_arguments()
    config = Dict(config)
    main(config)
