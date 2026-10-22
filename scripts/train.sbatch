#!/bin/bash
# SLURM job used for the reported run (1x V100 16GB, ~2h for 10 epochs).
# Set --account / --partition for your cluster, then submit from the repo root:
#   mkdir -p logs && sbatch scripts/train.sbatch
#SBATCH --job-name=blip_landmark_train
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.log

python src/train.py --data-root data/landmarks --output-dir checkpoints
