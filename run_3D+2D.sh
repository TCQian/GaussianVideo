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

MODEL_NAME_3D="GaussianVideo"
NUM_POINTS_3D=2500
TRAIN_ITERATIONS_3D=20000
QUANT_ITERATIONS_3D=10000
LEARNING_RATE_3D=0.01

MODEL_NAME_2D="GaussianImage_Cholesky"
NUM_POINTS_2D=750
TRAIN_ITERATIONS_2D=20000
QUANT_ITERATIONS_2D=10000

# Parse command-line arguments.
# Usage: ./script.sh --data_name MyData --num_points 30000 --start_frame 40 --num_frames 15
while [ "$#" -gt 0 ]; do
    case $1 in
        -d|--data_name)
            DATA_NAME="$2"
            shift 2
            ;;
        -p3d|--num_points_3d)
            NUM_POINTS_3D="$2"
            shift 2
            ;;
        -p2d|--num_points_2d)
            NUM_POINTS_2D="$2"
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

echo "Starting ${MODEL_NAME_3D}_${NUM_POINTS_3D}_${MODEL_NAME_2D}_${NUM_POINTS_2D}_${DATA_NAME}_${NUM_FRAMES}..."

# Define dataset and checkpoint paths using the variables.
YUV_PATH="/home/e/e0407638/github/GaussianVideo/YUV/${DATA_NAME}_1920x1080_120fps_420_8bit_YUV.yuv"
DATASET_PATH="/home/e/e0407638/github/GaussianVideo/dataset/${DATA_NAME}/"
CHECKPOINT_DIR_PATH="/home/e/e0407638/github/GaussianVideo/checkpoints/${DATA_NAME}/${MODEL_NAME_3D}_i${TRAIN_ITERATIONS_3D}_g${NUM_POINTS_3D}_${MODEL_NAME_2D}_i${TRAIN_ITERATIONS_2D}_g${NUM_POINTS_2D}_f${NUM_FRAMES}_s${START_FRAME}/"
CHECKPOINT_PATH_3D="${CHECKPOINT_DIR_PATH}${MODEL_NAME_3D}_i${TRAIN_ITERATIONS_3D}_g${NUM_POINTS_3D}_f${NUM_FRAMES}_s${START_FRAME}/"
CHECKPOINT_PATH_2D="${CHECKPOINT_DIR_PATH}${MODEL_NAME_2D}_${TRAIN_ITERATIONS_2D}_${NUM_POINTS_2D}/"
CHECKPOINT_QUANR_DIR_PATH="/home/e/e0407638/github/GaussianVideo/checkpoints_quant/${DATA_NAME}/${MODEL_NAME_3D}_i${QUANT_ITERATIONS_3D}_g${NUM_POINTS_3D}_${MODEL_NAME_2D}_i${QUANT_ITERATIONS_2D}_g${NUM_POINTS_2D}_f${NUM_FRAMES}_s${START_FRAME}/"
CHECKPOINT_QUANT_PATH_3D="${CHECKPOINT_QUANR_DIR_PATH}${MODEL_NAME_3D}_i${QUANT_ITERATIONS_3D}_g${NUM_POINTS_3D}_f${NUM_FRAMES}_s${START_FRAME}/"
CHECKPOINT_QUANT_PATH_2D="${CHECKPOINT_QUANR_DIR_PATH}${MODEL_NAME_2D}_${QUANT_ITERATIONS_2D}_${NUM_POINTS_2D}/"

python utils.py "${YUV_PATH}" --width 1920 --height 1080 --start_frame ${START_FRAME}

# Run the training script with the required arguments.
python train_3D+2D.py \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --iterations_3d "${TRAIN_ITERATIONS_3D}" \
    --model_name_3d "${MODEL_NAME_3D}" \
    --num_points_3d "${NUM_POINTS_3D}" \
    --lr_3d "${LEARNING_RATE_3D}" \
    --iterations_2d "${TRAIN_ITERATIONS_2D}" \
    --model_name_2d "${MODEL_NAME_2D}" \
    --num_points_2d "${NUM_POINTS_2D}" \
    --save_imgs

# Run the quantization training script.
python train_quantize_3D+2D.py \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --iterations_2d "${QUANT_ITERATIONS_2D}" \
    --model_name_2d "${MODEL_NAME_2D}" \
    --num_points_2d "${NUM_POINTS_2D}" \
    --model_path_2d "${CHECKPOINT_PATH_2D}" \
    --iterations_3d "${QUANT_ITERATIONS_3D}" \
    --model_name_3d "${MODEL_NAME_3D}" \
    --num_points_3d "${NUM_POINTS_3D}" \
    --model_path_3d "${CHECKPOINT_PATH_3D}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --lr_3d "${LEARNING_RATE_3D}" \
    --save_imgs

# # Run the quantization testing script.
python test_quantize_3D+2D.py \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --iterations_3d "${QUANT_ITERATIONS_3D}" \
    --model_name_3d "${MODEL_NAME_3D}" \
    --num_points_3d "${NUM_POINTS_3D}" \
    --model_path_3d "${CHECKPOINT_QUANT_PATH_3D}" \
    --iterations_2d "${QUANT_ITERATIONS_2D}" \
    --model_name_2d "${MODEL_NAME_2D}" \
    --num_points_2d "${NUM_POINTS_2D}" \
    --model_path_2d "${CHECKPOINT_QUANT_PATH_2D}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --lr_3d "${LEARNING_RATE_3D}" \
    --save_imgs

echo "Done"