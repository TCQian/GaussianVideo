#!/bin/sh
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G

echo "Starting..."

# Default variable values.
DATA_NAME="Beauty"
MODEL_NAME="GaussianVideo"
TRAIN_ITERATIONS=20000
QUANT_ITERATIONS=10000
LEARNING_RATE=0.01

# Default values for parameters to be overridden.
NUM_POINTS=10000
START_FRAME=0
NUM_FRAMES=5

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
DATASET_PATH="/home/l/leejiayi/GaussianVideo/dataset/${DATA_NAME}/"
CHECKPOINT_PATH="/home/l/leejiayi/GaussianVideo/checkpoints/${DATA_NAME}/${MODEL_NAME}_i${TRAIN_ITERATIONS}_g${NUM_POINTS}_f${NUM_FRAMES}_s${START_FRAME}/"
CHECKPOINT_QUANT_PATH="/home/l/leejiayi/GaussianVideo/checkpoints_quant/${DATA_NAME}/${MODEL_NAME}_i${QUANT_ITERATIONS}_g${NUM_POINTS}_f${NUM_FRAMES}_s${START_FRAME}/"

# Run the training script with the required arguments.
python train_video.py \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --iterations "${TRAIN_ITERATIONS}" \
    --model_name "${MODEL_NAME}" \
    --num_points "${NUM_POINTS}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --lr "${LEARNING_RATE}" \
    --save_imgs

# Run the quantization training script.
python train_quantize_video.py \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --iterations "${QUANT_ITERATIONS}" \
    --model_name "${MODEL_NAME}" \
    --num_points "${NUM_POINTS}" \
    --model_path "${CHECKPOINT_PATH}${DATA_NAME}/gaussian_model.pth.tar" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --lr "${LEARNING_RATE}" \
    --save_imgs

# Run the quantization testing script.
python test_quantize_video.py \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --iterations "${QUANT_ITERATIONS}" \
    --model_name "${MODEL_NAME}" \
    --num_points "${NUM_POINTS}" \
    --model_path "${CHECKPOINT_QUANT_PATH}${DATA_NAME}/gaussian_model.best.pth.tar" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --lr "${LEARNING_RATE}" \
    --save_imgs

echo "Done"