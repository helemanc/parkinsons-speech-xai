import os

import numpy as np
import torch
from addict import Dict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from yaml_config_override import add_arguments


class SpectrogramAttributionDataset(Dataset):
    def __init__(self, fold, model_dir, strategy):
        """
        Initialize dataset with data from a specified fold and strategy.
        Each sample contains an original spectrogram, attribution, saliency map, and label.
        """
        folder = os.path.join(model_dir, strategy)

        # Load data for the specified fold
        self.attribution = torch.load(
            os.path.join(folder, "attributions", f"fold_{fold}", "attributions.pt"),
            weights_only=False,
        )
        self.labels = torch.load(
            os.path.join(folder, "attributions", f"fold_{fold}", "gold_labels.pt"),
            weights_only=False,
        )
        self.originals = torch.load(
            os.path.join(folder, "attributions", f"fold_{fold}", "originals.pt"),
            weights_only=False,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns:
            original: Original spectrogram
            attribution: Attribution map
            saliency_map: Saliency map derived from the attribution and original
            label: Target label
        """
        original = self.originals[idx]
        attribution = self.attribution[idx]
        label = self.labels[idx]

        # Normalize the attribution map to create a saliency map
        attr_np = attribution.numpy()  # Convert to NumPy for min/max operations
        mask_normalized = (attr_np - np.min(attr_np)) / (
            np.max(attr_np) - np.min(attr_np) + 1e-5
        )  # Add epsilon to avoid division by zero
        saliency_map = original * torch.from_numpy(
            mask_normalized
        )  # Element-wise multiplication to create saliency map

        return original, attribution.float(), saliency_map.float(), label


def stratified_split(dataset, train_ratio=0.7, val_ratio=0.15):
    """
    Performs a stratified split to balance classes in train, validation, and test sets.

    Parameters:
        dataset: Dataset to split.
        train_ratio: Proportion of the dataset to use for training.
        val_ratio: Proportion of the dataset to use for validation.

    Returns:
        train_indices: Indices for the training subset.
        val_indices: Indices for the validation subset.
        test_indices: Indices for the test subset.
    """
    labels = [
        dataset[i][3].item() for i in range(len(dataset))
    ]  # Extract labels for stratification

    # First split into train+val and test
    train_val_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=1 - (train_ratio + val_ratio),
        stratify=labels,
        random_state=42,
    )

    # Extract labels for the train+val split to further split it into train and val
    train_val_labels = [labels[i] for i in train_val_indices]

    # Split train+val into train and validation
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio / (train_ratio + val_ratio),
        stratify=train_val_labels,
        random_state=42,
    )

    return train_indices, val_indices, test_indices


def get_stratified_dataloaders_for_fold(fold, config, model_dir, strategy):
    """
    Returns DataLoaders for train, validation, and test splits with stratified sampling.

    Parameters:
        fold: The current fold number.
        config: Configuration object containing batch size, etc.
        model_dir: Directory where data is stored.
        strategy: The attribution strategy being used.

    Returns:
        train_loader: DataLoader for the training subset.
        val_loader: DataLoader for the validation subset.
        test_loader: DataLoader for the test subset.
    """
    dataset = SpectrogramAttributionDataset(fold, model_dir, strategy)

    # Perform stratified split
    train_indices, val_indices, test_indices = stratified_split(
        dataset, train_ratio=0.7, val_ratio=0.15
    )

    # Create subsets using the stratified indices
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create DataLoaders for each split
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    return train_loader, val_loader, test_loader


# Example usage for all folds:

# def main(config):
#     model_dir = config.model.model_dir
#     strategy = config.training.strategy_key
#     model_dir = os.path.join("results", strategy)

#     for test_fold in range(1, 11):
#         dataloader = get_dataloader_for_fold(test_fold, config, model_dir, strategy)
#         print(f"Fold {test_fold} - Number of samples: {len(dataloader.dataset)}")

# if __name__ == "__main__":
#     config = add_arguments()
#     config = Dict(config)
#     main(config)
