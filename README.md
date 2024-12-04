# Investigating the Effectiveness of Explainability Methods in Parkinson’s Detection from Speech 🧠
This project focuses on evaluating interpretability methods for speech-based Parkinson's disease (PD) detection, aiming to identify PD-specific speech features and support clinical decision-making. While prior work has demonstrated strong classifier performance, this study emphasizes understanding and improving model explanations.

The project builds upon the classifier model proposed in [SSL4PR: Self-supervised Learning for Parkinson's Recognition](https://github.com/K-STMLab/SSL4PR/), using the PC-GITA dataset and 10-fold cross-validation for evaluation. The interpretability methods are assessed through:

1. **Attributions and Saliency Maps**: Applying mainstream explainability techniques to obtain explanations for PD-specific features.
2. **Faithfulness Evaluation**: Quantitatively evaluating the faithfulness of individual and combined saliency maps using established metrics.
3. **Auxiliary Classifier Assessment**: Analyzing the information conveyed by the explanations for PD detection using an auxiliary classifier.

Our results highlight that while the explanations align with the classifier’s predictions, they often lack sufficient utility for domain experts, underscoring the need for improved interpretability techniques in this domain.

**Table of Contents**
- [Setup](#setup)
- [Dataset and Data Splits](#dataset-and-data-splits)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Experiments](#experiments)
  - [Train SSL4PR model](#train-ssl4pr-model)
  - [Compute Explanations](#compute-explanations)
  - [Train CNN14 Classifier to Evaluate Explanations](#train-cnn14-classifier-to-evaluate-explanations)
- [Citation](#citation)



### Setup 
The project is based on Python 3.11 and PyTorch 2. The following command can be used to install the dependencies:

```bash
pip install -r requirements.txt
```

By default, the project leverages `comet_ml` for logging. To use it, you need to create an account on [comet.ml](https://www.comet.ml/). Then, you need to set the following environment variables:

```bash
export COMET_API_KEY=<your_api_key>
export COMET_WORKSPACE=<your_project_name>
```

**Disable logging**: 
To disable the logging, you can set the `training.use_comet` key to `false` in the `configs/*.yaml` files.

### Dataset and Data Splits

To make the results reproducible and comparable with the ones reported in the paper we make available the data splits used for 10-fold cross-validation. The splits are available in the `pcgita_splits` folder. The data is organized as follows:

```
pcgita_splits/
├── TRAIN_TEST_1
│   ├── test.csv
│   └── train.csv
├── TRAIN_TEST_2
│   ├── test.csv
│   └── train.csv
├── ...
└── TRAIN_TEST_10
    ├── test.csv
    └── train.csv
```

The `train.csv` and `test.csv` files contain the list of the audio files used for training and testing, respectively. The path to the audio files is stored in the following format:

```
/PC_GITA_ROOT_PATH/monologue/sin_normalizar/pd/AVPEPUDEA0042-Monologo-NR.wav
```

where `PC_GITA_ROOT_PATH` is the root path to the PC-GITA dataset. We also provide a python script `set_root_path.py` to set the root path to the PC-GITA dataset. The script can be used as follows:

```bash
python set_root_path.py --old_root_path <old_root_path> --new_root_path <new_root_path>
```

where `<old_root_path>` can be set to `PC_GITA_ROOT_PATH` and `<new_root_path>` is the new root path to your instance of the PC-GITA dataset.

**Note**: The splits are generated to ensure that the same speaker does not appear in both the training and testing sets and the classes are balanced across the splits.


### Exploratory Data Analysis 
To explore the structure of the data, you can run the following command:

```bash
python eda.py
```
This will generate visualizations of the analyzed data and distributions. The results will be saved in the `results/eda` folder. You can use these visualizations to better understand the characteristics of the dataset and the distribution of different features.

### Experiments

#### Train SSL4PR model 
To train the proposed model it is first needed to install the requirements and set the root path to the PC-GITA dataset. Then, the following command can be used to train the model:

```bash
python train.py --config <config_file> 
```

where `<config_file>` is the path to the configuration file (e.g., `configs/W_config.yaml`). There are several configuration parameter that can be set in the configuration file. The one reported in the paper is `configs/W_config.yaml`.

The training file creates 10 models, one for each fold, and compute the metrics for each fold. The metrics are reported on terminal and logged on comet.ml (if activated). Please be sure to update the following path in the configuration file:

```yaml
training:
  checkpoint_path: <path_to_save_checkpoints>
data:
  fold_root_path: <path_to_pcgita_splits>
```

where `<path_to_save_checkpoints>` is the path where the checkpoints will be saved and `<path_to_pcgita_splits>` is the path to the `pcgita_splits` folder.

**Pre-trained** models are available on the Hugging Face model hub ([SSL4PR WavLM Base](https://huggingface.co/morenolq/SSL4PR-wavlm-base), [SSL4PR HuBERT Base](https://huggingface.co/morenolq/SSL4PR-hubert-base)). To use them, please clone the repository running the following command:

```bash
# SSL4PR WavLM Base
git clone https://huggingface.co/morenolq/SSL4PR-wavlm-base

# SSL4PR HuBERT Base
git clone https://huggingface.co/morenolq/SSL4PR-hubert-base
```

Ensure you have `git lfs` installed. Each repository contains the pre-trained models, one per fold, named `fold_1.pt`, `fold_2.pt`, ..., `fold_10.pt`.

#### Compute Explanations 
To compute the saliency maps, run the following command:
```bash
python inference_only.py \
    --config configs/W_config.yaml \
    --training.checkpoint_path <CHECKPOINT_PATH> \
    --model.model_name_or_path <MODEL_NAME_OR_PATH> \
    --data.fold_root_path <FOLD_ROOT_PATH> \
    --training.strategy_key <STRATEGY_KEY> \
    --overlap False \
    --overlap_strategy <OVERLAP_STRATEGY>
```
where: 

- `--training.checkpoint_path`: Path to the directory containing pre-trained model checkpoints.  
  Example: `/path/to/checkpoints/SSL4PR-hubert-base/`

- `--model.model_name_or_path`: Specifies the base model to use for computing explanations.  
  Example: `facebook/hubert-base-ls960`

- `--data.fold_root_path`: Path to the root folder containing the dataset splits.  
  Example: `/path/to/pcgita_splits/`

- `--training.strategy_key`: Key to specify the explanation method to use (e.g., `ggc`).

- `--overlap`: Set to `True` to compute the overlap metrics for the explanations. Set to `False` produces explanations for the single methods (saliency maps, attributions, etc.).

- `--overlap_strategy`: Specifies the combination strategy for calculating overlap.  
  Can be either `sum` or `prod` (e.g., `sum` or `prod`).

**Output Information**
During this process, the results are stored in a `results` folder. If the folder does not already exist, it will be created automatically. The computed explanations are saved in subfolders named after the model used (e.g., `hubert` or `wavlm`).

**Faithfulness Metrics**
To compute the metrics on the overlap, the `--overlap` flag must be set to `True`, and the `--overlap_strategy` can be set to either `sum` or `prod`. This will allow the calculation of faithfulness metrics, specifically focusing on the intersection-over-union (IoU) between explanations from different methods. 

**Important Notes**
***Running Overlap***
The `--overlap True` flag will compute explanations for the single methods (such as saliency maps, attributions, etc.) and will combine them based on the specified overlap strategy (`sum` or `prod`).

***Compute Single Explanations First***
To run the overlap calculation, you must first compute the individual attributions, explanations, or saliency maps. The overlap strategy works on top of the previously computed explanations.



#### Train CNN14 Classifier to Evaluate Explanations 
To evaluate the saliency maps (i.e. explanations), we employ a CNN14 classifier, taking log-mel spectrograms as input. This classifier is trained for a binary classification task with the following hyperparameters:

- **Batch size**: 32  
- **Learning rate**: 0.002  
- **Training epochs**: 50  

We utilize the [SpeechBrain 1.0](https://speechbrain.github.io/) toolkit to train the classifier. For each fold, we construct a dataset from the saliency maps computed on the original test set. The dataset is split into training, validation, and test sets in a 70/15/15 ratio, ensuring balanced classes across splits. For comparison, the CNN14 classifier is also trained on the original spectrograms from the test set to evaluate performance differences between saliency map-based inputs and unmodified spectrograms.

**Training Instructions**

To train the CNN14 classifier, use the following command:

```bash
python train_cnn14.py configs/cnn14.yaml \
    --checkpoint_path <CHECKPOINT_PATH> \
    --model_name_or_path <MODEL_NAME_OR_PATH> \
    --fold_root_path <FOLD_ROOT_PATH> \
    --strategy_key <STRATEGY_KEY> \
    --batch_size 32 \
    --number_of_epochs 50 \
    --seed 42 \
    --use_masks=<USE_MASKS>
  
 ```

 where:
  - `--checkpoint_path`: Path to the directory containing pre-trained model checkpoints.  
  Example: `/path/to/checkpoints/SSL4PR-hubert-base/`  

- `--model_name_or_path`: Specifies the base model to use.  
  Example: `facebook/hubert-base-ls960`  

- `--fold_root_path`: Path to the root folder containing the dataset splits.  
  Example: `/path/to/pcgita_splits/`  

- `--strategy_key`: Key to specify the explanation method (e.g., `saliency`).  

- `--use_masks`: If set to `True`, the classifier takes saliency maps as input.  
  If set to `False`, it uses the original spectrograms.  

Ensure you use the configuration file located at `configs/cnn14.yaml` for the training setup.  

**N.B.** Training on saliency maps is possible only if the saliency maps have already been computed and stored in the `results` folder. To compute the explanations, please refer to the [Compute Explanations](#compute-explanations) section.


###  Citation

If you use this code, results from this project or you want to refer to the paper, please cite the following paper:

```bibtex
@article{mancini2024investigating,
  title = {Investigating the Effectiveness of Explainability Methods in Parkinson's Detection from Speech},
  author = {Mancini, Eleonora and Paissan, Francesco and Torroni, Paolo and Ravanelli, Mirco and Subakan, Cem},
  journal = {arXiv preprint arXiv:2411.08013},
  year = {2024}
}
```

<!-- # SSL4PR: Self-supervised learning for Parkinson's Recognition

This project aims at creating a DL system for Parkinson's recognition from speech. It leverages self-supervised learning models to transfer knowledge acquired by foundational models to the task of Parkinson's recognition. The project is based on the PC-GITA dataset and the proposed model is evaluated using 10-fold cross-validation.

**Table of Contents**
- [Setup](#setup)
- [Dataset and Data Splits](#dataset-and-data-splits)
- [Experiments](#experiments)
- [Citation](#citation)

### Setup

The project is based on Python 3.11 and PyTorch 2. The following command can be used to install the dependencies:

```bash
pip install -r requirements.txt
```

By default, the project leverages `comet_ml` for logging. To use it, you need to create an account on [comet.ml](https://www.comet.ml/). Then, you need to set the following environment variables:

```bash
export COMET_API_KEY=<your_api_key>
export COMET_WORKSPACE=<your_project_name>
```

**Disable logging**: 
To disable the logging, you can set the `training.use_comet` key to `false` in the `configs/*.yaml` files.

### Dataset and Data Splits

To make the results reproducible and comparable with the ones reported in the paper we make available the data splits used for 10-fold cross-validation. The splits are available in the `pcgita_splits` folder. The data is organized as follows:

```
pcgita_splits/
├── TRAIN_TEST_1
│   ├── test.csv
│   └── train.csv
├── TRAIN_TEST_2
│   ├── test.csv
│   └── train.csv
├── ...
└── TRAIN_TEST_10
    ├── test.csv
    └── train.csv
```

The `train.csv` and `test.csv` files contain the list of the audio files used for training and testing, respectively. The path to the audio files is stored in the following format:

```
/PC_GITA_ROOT_PATH/monologue/sin_normalizar/pd/AVPEPUDEA0042-Monologo-NR.wav
```

where `PC_GITA_ROOT_PATH` is the root path to the PC-GITA dataset. We also provide a python script `set_root_path.py` to set the root path to the PC-GITA dataset. The script can be used as follows:

```bash
python set_root_path.py --old_root_path <old_root_path> --new_root_path <new_root_path>
```

where `<old_root_path>` can be set to `PC_GITA_ROOT_PATH` and `<new_root_path>` is the new root path to your instance of the PC-GITA dataset.

**Note**: The splits are generated to ensure that the same speaker does not appear in both the training and testing sets and the classes are balanced across the splits.

### Experiments

To train the proposed model it is first needed to install the requirements and set the root path to the PC-GITA dataset. Then, the following command can be used to train the model:

```bash
python train.py --config <config_file> 
```

where `<config_file>` is the path to the configuration file (e.g., `configs/W_config.yaml`). There are several configuration parameter that can be set in the configuration file. The one reported in the paper is `configs/W_config.yaml`.

The training file creates 10 models, one for each fold, and compute the metrics for each fold. The metrics are reported on terminal and logged on comet.ml (if activated). Please be sure to update the following path in the configuration file:

```yaml
training:
  checkpoint_path: <path_to_save_checkpoints>
data:
  fold_root_path: <path_to_pcgita_splits>
```

where `<path_to_save_checkpoints>` is the path where the checkpoints will be saved and `<path_to_pcgita_splits>` is the path to the `pcgita_splits` folder.

**Pre-trained** models are available on the Hugging Face model hub ([SSL4PR WavLM Base](https://huggingface.co/morenolq/SSL4PR-wavlm-base), [SSL4PR HuBERT Base](https://huggingface.co/morenolq/SSL4PR-hubert-base)). To use them, please clone the repository running the following command:

```bash
# SSL4PR WavLM Base
git clone https://huggingface.co/morenolq/SSL4PR-wavlm-base

# SSL4PR HuBERT Base
git clone https://huggingface.co/morenolq/SSL4PR-hubert-base
```

Ensure you have `git lfs` installed. Each repository contains the pre-trained models, one per fold, named `fold_1.pt`, `fold_2.pt`, ..., `fold_10.pt`.

#### Inference on extended dataset

The paper presents the results of the proposed model on an extended dataset. The extended dataset is available in the `extended_dataset` folder. The script `infer_extended.py` can be used to infer the model on the extended dataset. The script can be used as follows:

```bash
python infer_extended.py --config <config_file> --training.ext_model_path <path_to_model> --ext_root_path <path_to_extended_dataset>
```

where `<config_file>` is the path to the configuration file (e.g., `configs/W_config.yaml`), `<path_to_model>` is the path to the model checkpoint and `<path_to_extended_dataset>` is the path to the `extended_dataset` folder.

**Speech Enhancement**: The proposed model can be used on the extended dataset in combination with speech enhancement preprocessing. To preprocess data following the same process as in the paper, the `speech_enhancement` folder contains the instructions to apply, VAD, dereverberation and noise reduction to the audio files.

### Citation

If you use this code, results from this project or you want to refer to the paper, please cite the following paper:

```
Will be available upon INTERSPEECH 2024 proceedings.
```
 -->
