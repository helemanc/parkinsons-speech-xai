from datasets import attribution_dataset
from yaml_config_override import add_arguments
from addict import Dict
import os 


def main(config):
    model_dir = config.model.model_dir
    strategy = config.training.strategy_key
    # Pretrained model
    pretrained_model = config.model.model_name_or_path

    if "wavlm" in pretrained_model:
        pretrained_model = "wavlm"
    elif "hubert" in pretrained_model:
        pretrained_model = "hubert"

    model_dir = os.path.join("results", pretrained_model)

    for test_fold in range(1, 11):
        dataloader = attribution_dataset.get_dataloader_for_fold(test_fold, config, model_dir, strategy)
        print(f"Fold {test_fold} - Number of samples: {len(dataloader.dataset)}")

if __name__ == "__main__":
    config = add_arguments()
    config = Dict(config)
    main(config)
