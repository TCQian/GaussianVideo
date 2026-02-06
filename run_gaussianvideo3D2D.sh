#!/bin/bash
#SBATCH --job-name=GaussianVideo    # Job name
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL                  # Get email for all status updates
#SBATCH --mail-user=e0407638@u.nus.edu   # Email for notifications

source ~/.bashrc
conda activate gv_h100
# cd gsplat
# pip install .[dev]
# cd ..

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
CHECKPOINT_DIR_PATH_QUANT="/home/e/e0407638/github/GaussianVideo/checkpoints_quant/${DATA_NAME}/ProgressiveGaussianVideo_i${TRAIN_ITERATIONS}_g${NUM_POINTS}_f${NUM_FRAMES}_s${START_FRAME}"

CHECKPOINT_PATH_LAYER0="${CHECKPOINT_DIR_PATH}/layer0/layer_0_model.pth.tar"
CHECKPOINT_PATH_LAYER1_GV3D2D="${CHECKPOINT_DIR_PATH}/layer1/GV3D2D_i${TRAIN_ITERATIONS}_g${NUM_POINTS}/layer_1_model.pth.tar"
CHECKPOINT_PATH_LAYER1_GVGI="${CHECKPOINT_DIR_PATH}/layer1/GVGI_i${TRAIN_ITERATIONS}_g${NUM_POINTS}/"

CHECKPOINT_QUANT_PATH_LAYER0="${CHECKPOINT_DIR_PATH_QUANT}/layer0/layer_0_model.best.pth.tar"
CHECKPOINT_QUANT_PATH_LAYER1_GV3D2D="${CHECKPOINT_DIR_PATH_QUANT}/layer1/GV3D2D_i${TRAIN_ITERATIONS}_g${NUM_POINTS}/layer_1_model.best.pth.tar"
CHECKPOINT_QUANT_PATH_LAYER1_GVGI="${CHECKPOINT_DIR_PATH_QUANT}/layer1/GVGI_i${TRAIN_ITERATIONS}_g${NUM_POINTS}/"

# Layer 0: GaussianVideo checkpoints (optional; use these as --model_path_layer0 to run 3D2D on top of GV)
GV_CHECKPOINT_DIR="/home/e/e0407638/github/GaussianVideo/checkpoints/${DATA_NAME}/GaussianVideo_i${TRAIN_ITERATIONS}_g${NUM_POINTS}_f${NUM_FRAMES}_s${START_FRAME}/${DATA_NAME}"
GV_CHECKPOINT_QUANT_DIR="/home/e/e0407638/github/GaussianVideo/checkpoints_quant/${DATA_NAME}/GaussianVideo_i${TRAIN_ITERATIONS}_g${NUM_POINTS}_f${NUM_FRAMES}_s${START_FRAME}/${DATA_NAME}"
CHECKPOINT_PATH_LAYER0_GV="${GV_CHECKPOINT_DIR}/gaussian_model.pth.tar"
CHECKPOINT_QUANT_PATH_LAYER0_GV="${GV_CHECKPOINT_QUANT_DIR}/gaussian_model.best.pth.tar"

# Run the training script with the required arguments.
# python train_3D2D.py \
#     --layer 0 \
#     --dataset "${DATASET_PATH}" \
#     --data_name "${DATA_NAME}" \
#     --start_frame "${START_FRAME}" \
#     --num_frames "${NUM_FRAMES}" \
#     --model_name "GV3D2D" \
#     --iterations "${TRAIN_ITERATIONS}" \
#     --num_points "${NUM_POINTS}" \
#     --lr "${LEARNING_RATE}" \
#     --save_imgs

python train_3D2D.py \
    --layer 1 \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --model_name "GV3D2D" \
    --iterations "${TRAIN_ITERATIONS}" \
    --num_points "${NUM_POINTS}" \
    --lr "${LEARNING_RATE}" \
    --model_path_layer0 "${CHECKPOINT_PATH_LAYER0_GV}" \
    --save_imgs

python train_3D2D.py \
    --layer 1 \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --model_name "GVGI" \
    --iterations "${TRAIN_ITERATIONS}" \
    --num_points "${NUM_POINTS}" \
    --lr "${LEARNING_RATE}" \
    --model_path_layer0 "${CHECKPOINT_PATH_LAYER0_GV}" \
    --save_imgs

# # Run the quantization training script.
# python train_quantize_3D2D.py \
#     --layer 0 \
#     --dataset "${DATASET_PATH}" \
#     --data_name "${DATA_NAME}" \
#     --start_frame "${START_FRAME}" \
#     --num_frames "${NUM_FRAMES}" \
#     --model_name "GV3D2D" \
#     --iterations "${TRAIN_ITERATIONS}" \
#     --num_points "${NUM_POINTS}" \
#     --lr "${LEARNING_RATE}" \
#     --model_path_layer0 "${CHECKPOINT_PATH_LAYER0}" \
#     --save_imgs

python train_quantize_3D2D.py \
    --layer 1 \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --model_name "GV3D2D" \
    --iterations "${TRAIN_ITERATIONS}" \
    --num_points "${NUM_POINTS}" \
    --lr "${LEARNING_RATE}" \
    --model_path_layer0 "${CHECKPOINT_QUANT_PATH_LAYER0_GV}" \
    --model_path_layer1 "${CHECKPOINT_PATH_LAYER1_GV3D2D}" \
    --save_imgs

python train_quantize_3D2D.py \
    --layer 1 \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --model_name "GVGI" \
    --iterations "${TRAIN_ITERATIONS}" \
    --num_points "${NUM_POINTS}" \
    --lr "${LEARNING_RATE}" \
    --model_path_layer0 "${CHECKPOINT_QUANT_PATH_LAYER0_GV}" \
    --model_path_layer1 "${CHECKPOINT_PATH_LAYER1_GVGI}" \
    --save_imgs


# # Run the quantization testing script. 
# #Only test with higher layer, the lower layer will be tested in the higher layer testing.
python test_quantize_3D2D.py \
    --layer 1 \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --model_name "GV3D2D" \
    --iterations "${TRAIN_ITERATIONS}" \
    --num_points "${NUM_POINTS}" \
    --lr "${LEARNING_RATE}" \
    --model_path_layer0 "${CHECKPOINT_QUANT_PATH_LAYER0_GV}" \
    --model_path_layer1 "${CHECKPOINT_QUANT_PATH_LAYER1_GV3D2D}" \
    --save_imgs

python test_quantize_3D2D.py \
    --layer 1 \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --start_frame "${START_FRAME}" \
    --num_frames "${NUM_FRAMES}" \
    --model_name "GVGI" \
    --iterations "${TRAIN_ITERATIONS}" \
    --num_points "${NUM_POINTS}" \
    --lr "${LEARNING_RATE}" \
    --model_path_layer0 "${CHECKPOINT_QUANT_PATH_LAYER0_GV}" \
    --model_path_layer1 "${CHECKPOINT_QUANT_PATH_LAYER1_GVGI}" \
    --save_imgs


echo "Done"