#!/bin/bash
#SBATCH --job-name=GaussianImage_HoneyBee          # Job name
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL                  # Get email for all status updates
#SBATCH --mail-user=e0407638@u.nus.edu   # Email for notifications

source ~/.bashrc
conda activate gv

echo "Starting..."

# Define variables for easy updating.
DATA_NAME="HoneyBee"
MODEL_NAME="GaussianImage_Cholesky"
NUM_POINTS=10000
TRAIN_ITERATIONS=20000
QUANT_ITERATIONS=10000

# Default values for parameters to be overridden.
NUM_POINTS=10000
START_FRAME=0
NUM_FRAMES=5

# Define dataset and checkpoint paths using the variables.
YUV_PATH="/home/e/e0407638/github/GaussianVideo/YUV/${DATA_NAME}_1920x1080_120fps_420_8bit_YUV.yuv"
DATASET_PATH="/home/e/e0407638/github/GaussianVideo/dataset/${DATA_NAME}/"
CHECKPOINT_PATH="/home/e/e0407638/github/GaussianVideo/checkpoints/${DATA_NAME}/${MODEL_NAME}_${TRAIN_ITERATIONS}_${NUM_POINTS}/"
CHECKPOINT_QUANT_PATH="/home/e/e0407638/github/GaussianVideo/checkpoints_quant/${DATA_NAME}/${MODEL_NAME}_${QUANT_ITERATIONS}_${NUM_POINTS}/"

python utils.py "${YUV_PATH}" --width 1920 --height 1080 --start_frame ${START_FRAME}

# Run the training script with the required arguments.
python train.py \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --iterations "${TRAIN_ITERATIONS}" \
    --model_name "${MODEL_NAME}" \
    --num_points "${NUM_POINTS}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --save_imgs

# Run the quantization training script.
python train_quantize.py \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --iterations "${QUANT_ITERATIONS}" \
    --model_name "${MODEL_NAME}" \
    --num_points "${NUM_POINTS}" \
    --model_path "${CHECKPOINT_PATH}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --save_imgs

# Run the quantization testing script.
python test_quantize.py \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --iterations "${QUANT_ITERATIONS}" \
    --model_name "${MODEL_NAME}" \
    --num_points "${NUM_POINTS}" \
    --model_path "${CHECKPOINT_QUANT_PATH}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --save_imgs

echo "Done"
