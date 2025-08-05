import math
from pathlib import Path
import argparse
import shutil
import yaml
import numpy as np
import torch
import sys
import random
import glob
import os
import cv2
import subprocess
import torch.nn.functional as F
from pytorch_msssim import ms_ssim as MS_SSIM

from utils import *
from train_video import images_paths_to_tensor
# Import common functions from train_3D+2D.py to avoid redundancy
import importlib.util
spec = importlib.util.spec_from_file_location("train_3d_2d", "train_3D+2D.py")
train_3d_2d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_3d_2d)

# Import the utility functions
get_delta_images = train_3d_2d.get_delta_images
combine_layers = train_3d_2d.combine_layers
evaluate_images = train_3d_2d.evaluate_images

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Test quantized 3D+2D GaussianVideo models.")
   
    # Parameters for testing
    parser.add_argument("--seed", type=int, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument("--quantize", action="store_true", help="Enable quantization")
    
    # Parameters for dataset
    parser.add_argument(
        "-d", "--dataset", type=str, default='./dataset/Jockey/', help="Test dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='Jockey', help="Test dataset"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=50,
        help="Number of frames (default: %(default)s)",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Start frame (default: %(default)s)",
    )

    # Model parameters for 3D GaussianVideo
    parser.add_argument(
        "--model_name_3d", type=str, default="GaussianVideo", help="model selection for 3D: GaussianVideo"
    )
    parser.add_argument(
        "--iterations_3d", type=int, default=50000, help="number of training epochs for 3D GaussianVideo (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points_3d",
        type=int,
        default=50000,
        help="2D+T GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path_3d", type=str, required=True, help="Path to a trained 3D GaussianVideo's checkpoint")
    parser.add_argument(
        "--lr_3d",
        type=float,
        default=1e-2,
        help="Learning rate of 3D GaussianVideo (default: %(default)s)",
    )

    # Model parameters for 2D GaussianImage
    parser.add_argument(
        "--iterations_2d", type=int, default=50000, help="number of training epochs for 2D GaussianImage (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name_2d", type=str, default="GaussianImage_Cholesky", help="model selection for 2D: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )
    parser.add_argument(
        "--num_points_2d",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path_2d", type=str, required=True, help="Path to a trained 2D GaussianImage's checkpoint")
    parser.add_argument(
        "--lr_2d",
        type=float,
        default=1e-3,
        help="Learning rate for 2D GaussianImage (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    test_dir_path = Path(f"./test_results_quant/{args.data_name}/{args.model_name_3d}_i{args.iterations_3d}_g{args.num_points_3d}_{args.model_name_2d}_i{args.iterations_2d}_g{args.num_points_2d}_f{args.num_frames}_s{args.start_frame}")
    os.makedirs(test_dir_path, exist_ok=True)
    logwriter = LogWriter(test_dir_path, train=False)
    
    # Testing 3D GaussianVideo as Layer 1
    image_length, start = args.num_frames, args.start_frame

    images_paths = []
    for i in range(start, start+image_length):
        image_path = Path(args.dataset) / f'frame_{i+1:04}.png'
        images_paths.append(image_path)

    logwriter.write(f"Testing quantized 3D GaussianVideo as Layer 1 with {args.num_frames} frames, {args.num_points_3d} points, model name: {args.model_name_3d}")
    
    cmd_args = [
        "python", "test_quantize_video.py",
        "--dataset", args.dataset,
        "--data_name", args.data_name,
        "--num_frames", str(args.num_frames),
        "--start_frame", str(args.start_frame),
        "--model_name_3d", args.model_name_3d,
        "--iterations_3d", str(args.iterations_3d),
        "--num_points_3d", str(args.num_points_3d),
        "--model_path_3d", args.model_path_3d,
        "--lr_3d", str(args.lr_3d),
        "--seed", str(args.seed)
    ]
    if args.save_imgs:
        cmd_args.append("--save_imgs")
    if args.quantize:
        cmd_args.append("--quantize")
        
    subprocess.run(cmd_args, check=True)
    logwriter.write("Quantized 3D GaussianVideo testing completed")

    # Find rendered images from 3D model test
    gaussianvideo_rendered_path = Path(f"./checkpoints_quant/{args.data_name}/{args.model_name_3d}_i{args.iterations_3d}_g{args.num_points_3d}_f{args.num_frames}_s{args.start_frame}/{args.data_name}")
    gaussianvideo_rendered_images = glob.glob(os.path.join(gaussianvideo_rendered_path, f"{args.data_name}_fitting_t*.png"))
    gaussianvideo_rendered_images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0][1:]))  # Sort by frame number

    # Generate delta images for 2D testing
    get_delta_images(images_paths, gaussianvideo_rendered_images, test_dir_path)

    # Testing 2D GaussianImage as Layer 2
    logwriter.write(f"Testing quantized 2D GaussianImage as Layer 2 with {args.num_points_2d} points, model name: {args.model_name_2d}")
    
    # Run test_quantize.py on delta images
    cmd_args = [
        "python", "test_quantize.py",
        "--dataset", os.path.join(test_dir_path, "delta_images"),
        "--data_name", args.data_name,
        "--iterations_2d", str(args.iterations_2d),
        "--model_name_2d", args.model_name_2d,
        "--num_points_2d", str(args.num_points_2d),
        "--model_path_2d", args.model_path_2d,
        "--lr_2d", str(args.lr_2d),
        "--num_frames", str(args.num_frames),
        "--start_frame", str(args.start_frame),
        "--seed", str(args.seed)
    ]
    if args.save_imgs:
        cmd_args.append("--save_imgs")
    if args.quantize:
        cmd_args.append("--quantize")
        
    subprocess.run(cmd_args, check=True)
    logwriter.write("Quantized 2D GaussianImage testing completed")

    # Find rendered images from 2D model test and combine layers
    gaussianimage_rendered_path = Path(f"./checkpoints_quant/{args.data_name}/{args.model_name_2d}_{args.iterations_2d}_{args.num_points_2d}")
    gaussianimage_rendered_images = glob.glob(os.path.join(gaussianimage_rendered_path, '*', f"frame_*_fitting.png"))
    gaussianimage_rendered_images.sort(key=lambda x: int(x.split('_')[-2]))  # Sort by frame number
    final_rendered_path = combine_layers(gaussianvideo_rendered_images, gaussianimage_rendered_images, test_dir_path)

    # Evaluate final combined results
    avg_psnr, avg_ms_ssim = evaluate_images(images_paths, final_rendered_path)
    logwriter.write("Final Test Results - PSNR:{:.4f}, MS-SSIM:{:.4f}".format(avg_psnr, avg_ms_ssim))

    # Copy important result folders for analysis
    result_3d_path = os.path.join(test_dir_path, "layer1_3d_results")
    result_2d_path = os.path.join(test_dir_path, "layer2_2d_results")
    
    if os.path.exists(gaussianvideo_rendered_path):
        shutil.copytree(gaussianvideo_rendered_path, result_3d_path, dirs_exist_ok=True)
    if os.path.exists(gaussianimage_rendered_path):
        shutil.copytree(gaussianimage_rendered_path, result_2d_path, dirs_exist_ok=True)
        
    logwriter.write(f"Test results saved to {test_dir_path}")

if __name__ == "__main__":
    main(sys.argv[1:]) 