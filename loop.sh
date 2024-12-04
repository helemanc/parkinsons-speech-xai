#!/bin/bash

for s in "saliency" "ig" "smoothgrad" "shap" "ggc" "gbp"
do
  CUDA_VISIBLE_DEVICES=4 python inference_only.py --config configs/W_config.yaml --training.checkpoint_path SSL4PR-hubert-base/ --model.model_name_or_path facebook/hubert-base-ls960 --data.fold_root_path pcgita_splits/ --training.strategy_key $s
done
