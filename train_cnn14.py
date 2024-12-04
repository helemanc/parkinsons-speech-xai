#!/usr/bin/python3

"""Recipe to train a classifier on ESC50 data.

To run this recipe, use the following command:
> python train.py hparams/<config>.yaml --data_folder yourpath/ESC-50-master

Authors
    * Cem Subakan 2022, 2023
    * Francesco Paissan 2022, 2023
    * Luca Della Libera 2024

Based on the Urban8k recipe by
    * David Whipps 2021
    * Ala Eddine Limame 2021
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchvision
#from confusion_matrix_fig import create_cm_fig
#from esc50_prepare import prepare_esc50
from hyperpyyaml import load_hyperpyyaml
from sklearn.metrics import confusion_matrix
#from wham_prepare import combine_batches, prepare_wham
from datasets import attribution_dataset

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main

import matplotlib.pyplot as plt
import librosa
from utils.viz import save_spectrograms

from tqdm import tqdm 
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader, SaveableDataLoader
from torcheval.metrics.functional import binary_f1_score as f1_score
import json 
from pathlib import Path

#mask = False # True if we want to train the model with mask, False otherwise
#save_specs = False # True if we want to save the spectrograms of the first batch




class ESC50Brain(sb.core.Brain):
    """Class for classifier training."""

    def compute_forward(self, batch, stage):
        """Computation pipeline based on an encoder + sound classifier."""
       
        batch = [item.to(self.device) for item in batch]

        original, mask , saliency_map, _ = batch #original and saliency map are log-magnitude spectrograms

        if self.hparams.use_masks:
            net_input = saliency_map
        else:
            net_input = original
        
        # If it's the first batch, save the spectrograms 
        #global save_specs  # Declare 'save_specs' as global to modify its value
        if self.hparams.save_specs:
            save_spectrograms(net_input, os.path.join(self.hparams.output_folder, "samples"))
            self.hparams.save_specs = False  # Set to False after the first batch

        if (
            hasattr(self.hparams, "use_melspectra")
            and self.hparams.use_melspectra
        ):
            net_input = self.modules.compute_fbank(net_input)

        # Embeddings + sound classifier
        if hasattr(self.modules.embedding_model, "config"):
            # Hugging Face model
            config = self.modules.embedding_model.config
            # Resize to match expected resolution
            net_input = torchvision.transforms.functional.resize(
                net_input, (config.image_size, config.image_size)
            )
            # Expand to have 3 channels
            net_input = net_input[:, None, ...].expand(-1, 3, -1, -1)
            if config.model_type == "focalnet":
                embeddings = self.modules.embedding_model(
                    net_input
                ).feature_maps[-1]
                embeddings = embeddings.mean(dim=(-1, -2))
            elif config.model_type == "vit":
                embeddings = self.modules.embedding_model(
                    net_input
                ).last_hidden_state.movedim(-1, -2)
                embeddings = embeddings.mean(dim=-1)
            else:
                raise NotImplementedError
        else:
            # SpeechBrain model
            embeddings = self.modules.embedding_model(net_input)
            if isinstance(embeddings, tuple):
                embeddings, _ = embeddings

            if embeddings.ndim == 4:
                embeddings = embeddings.mean((-1, -2))

        # run through classifier
        outputs = self.modules.classifier(embeddings)

        if outputs.ndim == 2:
            outputs = outputs.unsqueeze(1)

        return outputs#, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using class-id as label."""
        predictions = predictions
        #uttid = batch.id
        batch = [item.to(self.device) for item in batch]
        lens = len(batch[0])

        original, attribution, saliency_map, label = batch
        #classid, _ = batch.class_string_encoded
        classid = label.to(self.device)


        # Target augmentation
        N_augments = int(predictions.shape[0] / classid.shape[0])
        classid = torch.cat(N_augments * [classid], dim=0)
        
        target = classid.unsqueeze(1).float()

        loss_fn = sb.nnet.losses.bce_loss
        loss = loss_fn(predictions.squeeze(1).squeeze(1), target.squeeze(1), reduction="mean")

    
        if stage != sb.Stage.TEST:
            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

       
        # generate ids for the batch
        ids = [f"uttid_{i}" for i in range(len(classid))]
        preds = predictions.squeeze(1)
        preds_sigmoid = torch.sigmoid(preds)
        preds_sigmoid = preds_sigmoid.to(self.device)
        preds_sigmoid = (preds_sigmoid > 0.5).float()

        self.loss_metric.append(
            ids = ids, inputs=predictions.squeeze(1).squeeze(1), targets=target.squeeze(1), reduction="batch"
        )

        # Confusion matrices
        if stage != sb.Stage.TRAIN:
            y_true = classid.cpu().detach().numpy()
            y_pred = preds_sigmoid.cpu().detach().numpy()

        if stage == sb.Stage.VALID:
            confusion_mat = confusion_matrix(
                y_true,
                y_pred,
                labels= [0,1]
            )
            self.valid_confusion_matrix += confusion_mat
        if stage == sb.Stage.TEST:
            confusion_mat = confusion_matrix(
                y_true,
                y_pred,
                labels=[0,1],
            )
            self.test_confusion_matrix += confusion_mat

        # Compute accuracy using MetricStats
        self.acc_metric.append(
            torch.ones(lens), preds_sigmoid, classid.unsqueeze(1)
        )

        # if stage != sb.Stage.TRAIN:
        #     self.error_metrics.append(uttid, predictions, classid)# lens)
        # if stage != sb.Stage.TRAIN:
        #     self.error_metrics.append(ids, preds_sigmoid, classid, lens)# lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.bce_loss
        )

        # Compute accuracy using MetricStats
        # Define function taking (prediction, target, length) for eval
        def accuracy_value(predict, target):
            """Computes accuracy."""

            return torch.tensor([(predict == target).float().mean()])


        self.acc_metric = sb.utils.metric_stats.MetricStats(
            metric=accuracy_value, n_jobs=1
        )

        # Confusion matrices
        if stage == sb.Stage.VALID:
            self.valid_confusion_matrix = np.zeros(
                shape=(self.hparams.out_n_neurons+1, self.hparams.out_n_neurons+1), # +1 added for the case of binary classification
                dtype=int,
            )
        if stage == sb.Stage.TEST:
            self.test_confusion_matrix = np.zeros(
                shape=(self.hparams.out_n_neurons+1, self.hparams.out_n_neurons+1), # +1 added for the case of binary classification
                dtype=int,
            )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Compute/store important stats
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_stats = {
                "loss": self.train_loss,
                "acc": self.acc_metric.summarize("average"),
            }
        # Summarize Valid statistics from the stage for record-keeping
        elif stage == sb.Stage.VALID:
            valid_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                #"error": self.error_metrics.summarize("average"),
            }
        # Summarize Test statistics from the stage for record-keeping
        else:
            test_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                #"error": self.error_metrics.summarize("average"),
            }

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # Tensorboard logging
            if self.hparams.use_tensorboard:
                self.hparams.tensorboard_train_logger.log_stats(
                    stats_meta={"Epoch": epoch},
                    train_stats=self.train_stats,
                    valid_stats=valid_stats,
                )
                # # Log confusion matrix fig to tensorboard
                # cm_fig = create_cm_fig(
                #     self.valid_confusion_matrix,
                #     display_labels=list(
                #         self.hparams.label_encoder.ind2lab.values()
                #     ),
                # )
                # self.hparams.tensorboard_train_logger.writer.add_figure(
                #     "Validation Confusion Matrix", cm_fig, epoch
                # )

            # The train_logger writes a summary to stdout and to the log file
            self.hparams.train_log = os.path.join(self.hparams.output_folder, "train_log.txt")
            self.hparams.train_logger.save_file = os.path.join(self.hparams.output_folder, "train_log.txt")
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )
            # Save the current checkpoint and delete previous checkpoints,
            # self.checkpointer.save_and_keep_only(
            #     meta=valid_stats, min_keys=["error"]
            # )

        # We also write statistics about test data to stdout and to the log file
        if stage == sb.Stage.TEST: 
            # Per class accuracy from Test confusion matrix
            per_class_acc_arr = np.diag(self.test_confusion_matrix) / np.sum(
                self.test_confusion_matrix, axis=1
            )
            per_class_acc_arr_str = "\n" + "\n".join(
                "{:}: {:.3f}".format(class_id, class_acc)
                for class_id, class_acc in enumerate(per_class_acc_arr)
            )

            self.hparams.train_logger.log_stats(
                {
                    "Epoch loaded": self.hparams.epoch_counter.current,
                    "\n Per Class Accuracy": per_class_acc_arr_str,
                    "\n Confusion Matrix": "\n{:}\n".format(
                        self.test_confusion_matrix
                    ),
                },
                test_stats=test_stats,
            )
    
    def evaluate(self, test_set, max_key=None, min_key=None, progressbar=None, test_loader_kwargs={}):
        """Iterate test_set and evaluate brain performance, returning average test loss and accuracy."""
        
        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        # Check if test_set is a DataLoader; if not, create one
        if not (isinstance(test_set, DataLoader) or isinstance(test_set, LoopedLoader)):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(test_set, sb.Stage.TEST, **test_loader_kwargs)

        # Call on_evaluate_start to load the best checkpoint
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()

        # Initialize accumulators for loss and accuracy
        avg_test_loss = 0.0
        def accuracy_value(predict, target):
            """Computes accuracy."""

            return torch.tensor([(predict == target).float().mean()])
        
        def f1_score_value(predict, target):
            """Computes f1."""

            return torch.tensor([f1_score(predict, target)])

        def mask_mean(mask):
            return torch.tensor([mask.mean()])
        
        def mask_std(mask):
            return torch.tensor([mask.std()])
        
        def selective_accuracy(predict, target,  mask_mean): 
            accuracy = accuracy_value(predict, target)
            return accuracy * (1 - mask_mean)
        
        def selective_f1(predict, target,  mask_mean): 
            f1 = f1_score_value(predict, target)
            return f1 * (1 - mask_mean)

        self.acc_metric = sb.utils.metric_stats.MetricStats(
            metric=accuracy_value, n_jobs=1
        )

        self.f1_metric = sb.utils.metric_stats.MetricStats(
            metric=f1_score_value, n_jobs=1
        )

        self.mask_mean = sb.utils.metric_stats.MetricStats(
            metric=mask_mean, n_jobs=1
        )

        self.selective_accuracy = sb.utils.metric_stats.MetricStats(
            metric=selective_accuracy, n_jobs=1
        )

        self.selective_f1 = sb.utils.metric_stats.MetricStats(
            metric=selective_f1, n_jobs=1
        )

        self.mask_std = sb.utils.metric_stats.MetricStats(
            metric=mask_std, n_jobs=1
        )


        with torch.no_grad():
            for batch in tqdm(
                test_set,
                dynamic_ncols=True,
                disable=not enable,
                colour=self.tqdm_barcolor["test"],
            ):
                self.step += 1
                loss = self.evaluate_batch(batch, stage=sb.Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Append accuracy for each batch
                predictions  = self.compute_forward(batch, stage=sb.Stage.TEST)
                _, masks, _, label = batch
                preds = predictions.squeeze(1)
                preds_sigmoid = torch.sigmoid(preds)
                preds_sigmoid = preds_sigmoid.to(self.device)
                preds_sigmoid = (preds_sigmoid > 0.5).float().squeeze(1)
                mask_mean_val = mask_mean(masks)
                
                self.acc_metric.append(torch.ones(len(label)), preds_sigmoid.to(self.device), label.to(self.device))
                self.f1_metric.append(torch.ones(len(label)), preds_sigmoid.to(self.device), label.to(self.device))
                self.mask_mean.append(torch.ones(len(label)), masks)
                self.selective_accuracy.append(torch.ones(len(label)), preds_sigmoid.to(self.device), label.to(self.device), mask_mean_val)
                self.selective_f1.append(torch.ones(len(label)), preds_sigmoid.to(self.device), label.to(self.device), mask_mean_val)
                self.mask_std.append(torch.ones(len(label)), masks)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # Finalize stage (updates the confusion matrix and stats in on_stage_end)
            self.on_stage_end(sb.Stage.TEST, avg_test_loss, None)

        # Reset the step count and summarize accuracy
        self.step = 0
        avg_accuracy = self.acc_metric.summarize("average")
        avg_f1 = self.f1_metric.summarize("average")
        avg_mask_mean = self.mask_mean.summarize("average")
        avg_selective_accuracy = self.selective_accuracy.summarize("average")
        avg_selective_f1 = self.selective_f1.summarize("average")
        avg_mask_std = self.mask_std.summarize("average")

        return avg_test_loss, avg_accuracy, avg_f1, avg_mask_mean, avg_selective_accuracy, avg_selective_f1, avg_mask_std



if __name__ == "__main__":
    # This flag enables the built-in cuDNN auto-tuner
    # torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    print(hparams_file)
    print(run_opts)
    print(overrides)


    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
   
    strategy = hparams["strategy_key"]
    # Pretrained model
    pretrained_model = hparams["model_name_or_path"]

    if "wavlm" in pretrained_model:
        pretrained_model = "wavlm"
    elif "hubert" in pretrained_model:
        pretrained_model = "hubert"

    model_dir = os.path.join("results", pretrained_model)


    hparams["experiment_name"] = "cnn14"

    avg_accuracy_list = []
    avg_f1_list = []
    avg_mask_mean_list = []
    avg_selective_accuracy_list = []
    avg_selective_f1_list = []
    avg_mask_std_list = []

    
    for test_fold in range(1, 11):
        print("Training fold: ", test_fold) 
        train_dataloader, val_dataloader, test_dataloader = attribution_dataset.get_stratified_dataloaders_for_fold(test_fold, hparams, model_dir, strategy)
        print(f"Fold {test_fold} - Number of samples: {len(train_dataloader.dataset)}")
        print(f"Fold {test_fold} - Number of samples: {len(val_dataloader.dataset)}")
        print(f"Fold {test_fold} - Number of samples: {len(test_dataloader.dataset)}")
        
        if hparams["use_masks"]: 
            hparams["output_folder"] = os.path.join("results", hparams["experiment_name"], str(hparams["seed"]), pretrained_model, strategy, f"fold_{test_fold}")
        else:
            hparams["output_folder"] = os.path.join("results", hparams["experiment_name"], str(hparams["seed"]), pretrained_model, "originals", f"fold_{test_fold}")
       
        # hparams["save_folder"] = os.path.join(hparams["output_folder"], "save")
        # hparams['checkpointer'].checkpoints_dir= hparams["save_folder"]
        hparams["train_log"] = os.path.join(hparams["output_folder"], "train_log.txt")



        # Create experiment directory
        sb.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )

        # Tensorboard logging
        if hparams["use_tensorboard"]:
            from speechbrain.utils.train_logger import TensorboardLogger

            hparams["tensorboard_train_logger"] = TensorboardLogger(
                hparams["tensorboard_logs_folder"]
            )


        ESC50_brain = ESC50Brain(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            #checkpointer=hparams["checkpointer"],
        )

        # Load pretrained encoder if it exists in the yaml file
        if not hasattr(ESC50_brain.modules, "embedding_model"):
            ESC50_brain.hparams.embedding_model.to(ESC50_brain.device)

        if "pretrained_encoder" in hparams and hparams["use_pretrained"]:
            run_on_main(hparams["pretrained_encoder"].collect_files)
            hparams["pretrained_encoder"].load_collected()

        
        epoch_counter = sb.utils.epoch_loop.EpochCounter(limit=hparams["number_of_epochs"])


        if not hparams["test_only"]:
            ESC50_brain.fit(
                epoch_counter=epoch_counter,
                train_set=train_dataloader,
                valid_set=val_dataloader,
                train_loader_kwargs=hparams["dataloader_options"],
                valid_loader_kwargs=hparams["dataloader_options"],
            )
        


        # Load the best checkpoint for evaluation
        avg_test_loss, avg_accuracy, avg_f1, avg_mask_mean, avg_selective_accuracy, avg_selective_f1, avg_mask_std  = ESC50_brain.evaluate(
            test_set=test_dataloader,
            min_key="error",
            progressbar=True,
            test_loader_kwargs=hparams["dataloader_options"],
        )

        print("avg_test_loss: ", avg_test_loss)
        print("avg_accuracy: ", avg_accuracy)
        print("avg_f1: ", avg_f1)
        print("avg_mask_mean: ", avg_mask_mean)
        print("avg_selective_accuracy: ", avg_selective_accuracy)
        print("avg_selective_f1: ", avg_selective_f1)
        print("avg_mask_std: ", avg_mask_std)

        avg_accuracy_list.append(avg_accuracy)
        avg_f1_list.append(avg_f1)
        avg_mask_mean_list.append(avg_mask_mean)
        avg_selective_accuracy_list.append(avg_selective_accuracy)
        avg_selective_f1_list.append(avg_selective_f1)
        avg_mask_std_list.append(avg_mask_std)



        # Save the results under the save folder as jsons 
        with open(os.path.join(hparams["output_folder"], "results.json"), "w") as f:
            json.dump({"avg_test_loss": avg_test_loss, 
                            "avg_accuracy": avg_accuracy, 
                            "avg_f1": avg_f1,
                            "avg_mask_mean": avg_mask_mean,
                            "avg_selective_accuracy": avg_selective_accuracy,
                            "avg_selective_f1": avg_selective_f1,
                            "avg_mask_std": avg_mask_std}, f, indent=4)
    
    # Save the average results for all the foldsù
    if hparams["use_masks"]:
        file_name = os.path.join("results", hparams["experiment_name"], str(hparams["seed"]), pretrained_model, strategy, "avg_results.json")
    else:
        file_name = os.path.join("results", hparams["experiment_name"], str(hparams["seed"]), pretrained_model, "originals", "avg_results.json")
    with open(file_name, "w") as f:
        json.dump({"avg_accuracy": np.mean(avg_accuracy_list), 
                    "avg_f1": np.mean(avg_f1_list),
                    "avg_mask_mean": np.mean(avg_mask_mean_list),
                    "avg_selective_accuracy": np.mean(avg_selective_accuracy_list),
                    "avg_selective_f1": np.mean(avg_selective_f1_list),
                    "avg_mask_std": avg_mask_std}, f, indent=4)
            
