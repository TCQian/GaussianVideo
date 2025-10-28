#!/bin/bash
#SBATCH --job-name=GaussianVideo    # Job name
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL                  # Get email for all status updates
#SBATCH --mail-user=e0407638@u.nus.edu   # Email for notifications

source ~/.bashrc
conda activate gv_h100
cd gsplat
pip install .[dev]
cd ..

# Default variable values.
DATA_NAME="Beauty"
START_FRAME=0
NUM_FRAMES=5

NUM_POINTS=12000
TRAIN_ITERATIONS=50000
LEARNING_RATE=0.001

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

echo "Starting ProgressiveGaussianVideo_i${TRAIN_ITERATIONS}_g${NUM_POINTS}_f${NUM_FRAMES}_s${START_FRAME}..."

# Define dataset and checkpoint paths using the variables.
YUV_PATH="/home/e/e0407638/github/GaussianVideo/YUV/${DATA_NAME}_1920x1080_120fps_420_8bit_YUV.yuv"
DATASET_PATH="/home/e/e0407638/github/GaussianVideo/dataset/${DATA_NAME}/"
CHECKPOINT_DIR_PATH="/home/e/e0407638/github/GaussianVideo/checkpoints/${DATA_NAME}/ProgressiveGaussianVideo_i${TRAIN_ITERATIONS}_g${NUM_POINTS}_f${NUM_FRAMES}_s${START_FRAME}"
CHECKPOINT_PATH="${CHECKPOINT_DIR_PATH}/layer0/layer_0_model.pth.tar"

# Run the training script with the required arguments.
python gaussianvideo3D2D.py \
    --layer 0 \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --model_name "GV3D2D" \
    --iterations "${TRAIN_ITERATIONS}" \
    --num_points "${NUM_POINTS}" \
    --lr "${LEARNING_RATE}" \
    --save_imgs

python gaussianvideo3D2D.py \
    --layer 1 \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --model_name "GV3D2D" \
    --iterations "${TRAIN_ITERATIONS}" \
    --num_points "${NUM_POINTS}" \
    --lr "${LEARNING_RATE}" \
    --model_path "${CHECKPOINT_PATH}" \
    --save_imgs

python gaussianvideo3D2D.py \
    --layer 1 \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --model_name "GVGI" \
    --iterations "${TRAIN_ITERATIONS}" \
    --num_points "${NUM_POINTS}" \
    --lr "${LEARNING_RATE}" \
    --model_path "${CHECKPOINT_PATH}" \
    --save_imgs


echo "Done"