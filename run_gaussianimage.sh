#!/bin/sh
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G

echo "Starting..."

# Define variables for easy updating.
DATA_NAME="Beauty"
MODEL_NAME="GaussianImage_Cholesky"
NUM_POINTS=10000
TRAIN_ITERATIONS=20000
QUANT_ITERATIONS=10000

# Define dataset and checkpoint paths using the variables.
DATASET_PATH="/home/l/leejiayi/GaussianVideo/dataset/${DATA_NAME}/"
CHECKPOINT_PATH="/home/l/leejiayi/GaussianVideo/checkpoints/${DATA_NAME}/${MODEL_NAME}_${TRAIN_ITERATIONS}_${NUM_POINTS}/"
CHECKPOINT_QUANT_PATH="/home/l/leejiayi/GaussianVideo/checkpoints_quant/${DATA_NAME}/${MODEL_NAME}_${QUANT_ITERATIONS}_${NUM_POINTS}/"

# Run the training script with the required arguments.
python train.py \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --iterations "${TRAIN_ITERATIONS}" \
    --model_name "${MODEL_NAME}" \
    --num_points "${NUM_POINTS}" \
    --save_imgs

# Run the quantization training script.
python train_quantize.py \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --iterations "${QUANT_ITERATIONS}" \
    --model_name "${MODEL_NAME}" \
    --num_points "${NUM_POINTS}" \
    --model_path "${CHECKPOINT_PATH}" \
    --save_imgs

# Run the quantization testing script.
python test_quantize.py \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --iterations "${QUANT_ITERATIONS}" \
    --model_name "${MODEL_NAME}" \
    --num_points "${NUM_POINTS}" \
    --model_path "${CHECKPOINT_QUANT_PATH}" \
    --save_imgs

echo "Done"
