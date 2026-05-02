#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.

echo "CLEANING OLD DATA..."
rm -rf data/sft data/dpo
mkdir -p data/sft data/dpo

echo "STEP 1: Starting SFT Data Preparation (Balanced)..."
python3 scripts/prepare_sft_dataset.py > sft_prep.log 2>&1

echo "STEP 2: Starting SFT Training (Optimized LoRA)..."
# Using mlx_lm.lora with a config file for better rank/alpha control
python3 -m mlx_lm.lora \
    --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --train \
    --data data/sft \
    --iters 400 \
    --learning-rate 1e-5 \
    --batch-size 4 \
    --adapter-path adapters \
    --config trainers/mlx/lora_config.yaml >> train.log 2>&1

echo "STEP 3: Starting DPO Data Preparation (Fixed)..."
python3 scripts/prepare_dpo_dataset.py >> dpo_prep.log 2>&1

echo "STEP 4: Starting DPO Training (Fixed Loop)..."
python3 trainers/mlx/dpo_train_manual.py >> train.log 2>&1

echo "DONE."
