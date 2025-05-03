#!/bin/bash
#SBATCH --job-name=GaussianImage_${DATA_NAME}_${NUM_FRAMES}_${NUM_POINTS}          # Job name
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
NUM_FRAMES=1

# Parse command-line arguments.
# Usage: ./script.sh --data_name MyData --num_points 30000 --start_frame 40 --num_frames 15
while [ "$#" -gt 0 ]; do
    case $1 in
        -d|--data_name)
            DATA_NAME="$2"
            shift 2
            ;;
        -p|--num_points)
            NUM_POINTS="$2"
            shift 2
            ;;
        -s|--start_frame)
            START_FRAME="$2"
            shift 2
            ;;
        -n|--num_frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 [--data_name <value>] [--num_points <value>] [--start_frame <value>] [--num_frames <value>]"
            exit 1
            ;;
    esac
done

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
