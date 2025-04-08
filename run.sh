#!/bin/sh
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G

echo "Starting..."

srun git add .

echo "Done"