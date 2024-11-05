import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from yaml_config_override import add_arguments
from addict import Dict

class SpectrogramAttributionDataset(Dataset):
    def __init__(self, fold, model_dir, strategy):
        """
        Initialize dataset with data from a specified fold and strategy.
        Each sample contains an original spectrogram, attribution, saliency map, and label.
        """
        folder = os.path.join(model_dir, strategy)
        
        # Load data for the specified fold
        self.attribution = torch.load(os.path.join(folder, "attributions", f"fold_{fold}", "attributions.pt"))
        self.labels = torch.load(os.path.join(folder, "attributions", f"fold_{fold}", "gold_labels.pt"))
        self.originals = torch.load(os.path.join(folder, "attributions", f"fold_{fold}", "originals.pt"))

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
        mask_normalized = (attr_np - np.min(attr_np)) / (np.max(attr_np) - np.min(attr_np) + 1e-5)  # Add epsilon to avoid division by zero
        saliency_map = original * torch.from_numpy(mask_normalized)  # Element-wise multiplication to create saliency map
        
        return original, attribution, saliency_map, label


def get_dataloader_for_fold(fold, config, model_dir, strategy):
    """
    Returns DataLoader for a specific fold and strategy.
    """
    dataset = SpectrogramAttributionDataset(fold, model_dir, strategy)
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)
    return dataloader

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

