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
pip install .[dev] --no-build-isolation
cd ..

# Default variable values.
DATA_NAME="Beauty"
MODEL_NAME="GaussianVideo"
TRAIN_ITERATIONS=50000
QUANT_ITERATIONS=50000
LEARNING_RATE=0.001

# Default values for parameters to be overridden.
NUM_POINTS=30000
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

echo "Starting GaussianVideo_${DATA_NAME}_${NUM_FRAMES}_${NUM_POINTS}..."

# Define dataset and checkpoint paths using the variables.
YUV_PATH="/home/e/e0407638/github/GaussianVideo/YUV/${DATA_NAME}_1920x1080_120fps_420_8bit_YUV.yuv"
GV_CHECKPOINT_DIR="/home/e/e0407638/github/GaussianVideo/checkpoints/${DATA_NAME}/GaussianVideo_i${TRAIN_ITERATIONS}_g${NUM_POINTS}_f${NUM_FRAMES}_s${START_FRAME}/${DATA_NAME}"

python test.py --checkpoint ${GV_CHECKPOINT_DIR}/gaussian_model.pth.tar --H 1080 --W 1920 --T 5
python visualize_czz_spread.py --checkpoint ${GV_CHECKPOINT_DIR}/gaussian_model.pth.tar --H 1080 --W 1920 --T 5 --out_dir ./viz_czz_spread

echo "Done"