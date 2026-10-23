#!/bin/bash
# Evaluate the fine-tuned model and the zero-shot BLIP baseline.
# Submit from the repo root: mkdir -p logs && sbatch scripts/evaluate.sbatch
#SBATCH --job-name=blip_landmark_eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_%j.log

python src/evaluate.py --data-root data/landmarks --checkpoint checkpoints --split holdout --tag finetuned
python src/evaluate.py --data-root data/landmarks --split holdout --tag zeroshot
