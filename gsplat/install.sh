#!/bin/sh
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G

echo "Starting..."

# Do the job and tell it where to find the working directory.
# srun pip install -r requirements.txt
srun pip install .[dev]

# If srun completes successfully, the following will be printed
echo "Done"
