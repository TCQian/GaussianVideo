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

def get_delta_images(gt_images_paths, rendered_images_paths, output_path):
    assert len(rendered_images_paths) == len(gt_images_paths), f"Number of rendered images, {len(rendered_images_paths)}, does not match the number of frames, {len(gt_images_paths)}."

    delta_path = os.path.join(output_path, "delta_images")
    os.makedirs(delta_path, exist_ok=True)

    # mins, maxs = [], []
    for gt_img_path, rendered_img_path in zip(gt_images_paths, rendered_images_paths):
        print(f"Getting delta image for {gt_img_path} and {rendered_img_path}")
        gt_image = cv2.imread(str(gt_img_path), cv2.IMREAD_UNCHANGED).astype(np.int16) #/ 255.0
        rendered_image = cv2.imread(rendered_img_path, cv2.IMREAD_UNCHANGED).astype(np.int16) #/ 255.0
        delta_image = gt_image - rendered_image
        # delta_image += 1.0 # shift to 0-2
        # print the min and max values of the delta image
        min = np.min(delta_image) 
        max = np.max(delta_image)
        # mins.append(min)
        # maxs.append(max)
        print(f"Delta image min: {min}, max: {max}")
        delta_image = (delta_image + 255.0) / 2.0
        # Keep both -ve and +ve values in the delta image
        # delta_image = (delta_image - min) / (max - min)
        # delta_image = np.clip(delta_image * 255.0, 0, 255).astype(np.uint8)
        # Save the delta image
        delta_image_path = os.path.join(delta_path, os.path.basename(gt_img_path))
        cv2.imwrite(delta_image_path, delta_image)
    return delta_path #, mins, maxs

def combine_layers(layer1_images_paths, layer2_images_paths, output_path):
    assert len(layer1_images_paths) == len(layer2_images_paths), f"Number of Layer 1 images, {len(layer1_images_paths)}, does not match the number of Layer 2 images, {len(layer2_images_paths)}."

    final_rendered_path = os.path.join(output_path, "final_rendered")
    os.makedirs(final_rendered_path, exist_ok=True)
    for layer1_img, layer2_img in zip(layer1_images_paths, layer2_images_paths): #, mins, maxs):
        print(f"Combining {layer1_img} and {layer2_img}")
        layer1_image = cv2.imread(layer1_img, cv2.IMREAD_UNCHANGED).astype(np.int16) #/ 255.0
        layer2_image = cv2.imread(layer2_img, cv2.IMREAD_UNCHANGED).astype(np.int16) #/ 255.0
        # restore -ve and +ve values in the delta image
        # layer2_image = (layer2_image * (max - min)) + min - 1.0
        layer2_image = (layer2_image * 2.0) - 255.0
        print(f"Layer 2 image min: {np.min(layer2_image)}, max: {np.max(layer2_image)}")
        # Combine the two layers
        final_image = np.clip(layer1_image + layer2_image, 0, 255).astype(np.uint8)
        # final_image = np.clip((layer1_image + layer2_image) * 255.0, 0, 255).astype(np.uint8)
        final_image_name = os.path.basename(layer2_img)
        final_image_path = os.path.join(final_rendered_path, final_image_name)
        cv2.imwrite(final_image_path, final_image)
    print(f"Final rendered images saved to {final_rendered_path}")

    return final_rendered_path

def evaluate_images(gt_images_paths, output_images_paths):
    gt_images_tensor = images_paths_to_tensor(gt_images_paths)
    output_images_tensor = images_paths_to_tensor([os.path.join(output_images_paths, f"frame_{i+1:04}_fitting.png") for i in range(len(gt_images_paths))])
    mse_loss = F.mse_loss(output_images_tensor.float(), gt_images_tensor.float())
    avg_psnr = 10 * math.log10(1.0 / mse_loss.item())

    num_time_steps = output_images_tensor.size(-1)  # T dimension

    ms_ssim_values = []
    for t in range(num_time_steps):
        # Extract the t-th frame from both render and ground truth
        frame = output_images_tensor[..., t]  # e.g. shape: [1, 3, H, W]
        gt_frame = gt_images_tensor[..., t] # e.g. shape: [1, 3, H, W]
        ms_ssim_values.append(
            MS_SSIM(frame, gt_frame, data_range=1, size_average=True).item()
        )
    avg_ms_ssim = sum(ms_ssim_values) / len(ms_ssim_values)

    return avg_psnr, avg_ms_ssim

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
   
    # Parameters for training
    parser.add_argument("--seed", type=int, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    
    # Parameters for dataset
    parser.add_argument(
        "-d", "--dataset", type=str, default='./dataset/Jockey/', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='Jockey', help="Training dataset"
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
    parser.add_argument("--model_path_3d", type=str, default=None, help="Path to a 3D GaussianVideo's checkpoint")
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
    parser.add_argument("--model_path_2d", type=str, default=None, help="Path to a 2D GaussianImage's checkpoint")
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

    final_dir_path = Path(f"./checkpoints/{args.data_name}/{args.model_name_3d}_i{args.iterations_3d}_g{args.num_points_3d}_{args.model_name_2d}_i{args.iterations_2d}_g{args.num_points_2d}_f{args.num_frames}_s{args.start_frame}")
    logwriter = LogWriter(final_dir_path)
    
    # Training 3D GaussianVideo as Layer 1
    image_length, start = args.num_frames, args.start_frame

    images_paths = []
    for i in range(start, start+image_length):
        image_path = Path(args.dataset) / f'frame_{i+1:04}.png'
        images_paths.append(image_path)
        
    gaussianvideo_rendered_path = Path(f"./checkpoints/{args.data_name}/{args.model_name_3d}_i{args.iterations_3d}_g{args.num_points_3d}_f{args.num_frames}_s{args.start_frame}/{args.data_name}")
    gv_done = os.path.exists(gaussianvideo_rendered_path) and len(glob.glob(os.path.join(gaussianvideo_rendered_path, f"{args.data_name}_fitting_t*.png"))) == image_length
    if not gv_done:
        logwriter.write(f"Training 3D GaussianVideo as Layer 1 with {args.num_frames} frames, {args.num_points_3d} points, {args.iterations_3d} iterations, model name: {args.model_name_3d}")
        
        cmd_args = [
            "python", "train_video.py",
            "--dataset", args.dataset,
            "--data_name", args.data_name,
            "--num_frames", str(args.num_frames),
            "--start_frame", str(args.start_frame),
            "--model_name_3d", args.model_name_3d,
            "--iterations_3d", str(args.iterations_3d),
            "--num_points_3d", str(args.num_points_3d),
            "--lr_3d", str(args.lr_3d),
            "--seed", str(args.seed)
        ]
        if args.model_path_3d:
            cmd_args.extend(["--model_path_3d", args.model_path_3d])
        if args.save_imgs:
            cmd_args.append("--save_imgs")
            
        subprocess.run(cmd_args, check=True)
        logwriter.write("3D GaussianVideo training completed")

    # Collect delta image for 2D GaussianImage training
    gaussianvideo_rendered_images = glob.glob(os.path.join(gaussianvideo_rendered_path, f"{args.data_name}_fitting_t*.png"))
    gaussianvideo_rendered_images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0][1:]))  # Sort by frame number
    # os.makedirs(os.path.join(final_dir_path, "background"), exist_ok=True)
    # for i in range(len(gaussianvideo_rendered_images)): # copy rendered image to background folder
    #     shutil.copy(gaussianvideo_rendered_images[i], os.path.join(final_dir_path, "background", f"frame_{i+1:04}.png"))
    get_delta_images(images_paths, gaussianvideo_rendered_images, final_dir_path)

    # Training 2D GaussianImage as Layer 2
    logwriter.write(f"Training 2D GaussianImage as Layer 2 with {args.num_points_2d} points, {args.iterations_2d} iterations, model name: {args.model_name_2d}")
    
    # Run train.py once to process all delta images
    cmd_args = [
        "python", "train.py",
        "--dataset", args.dataset, #os.path.join(final_dir_path, "delta_images"),
        # "--background", os.path.join(final_dir_path, "background"),
        "--data_name", args.data_name,
        "--iterations_2d", str(args.iterations_2d),
        "--model_name_2d", args.model_name_2d,
        "--num_points_2d", str(args.num_points_2d),
        "--lr_2d", str(args.lr_2d),
        "--num_frames", str(args.num_frames),
        "--start_frame", str(args.start_frame),
        "--seed", str(args.seed)
    ]
    if args.model_path_2d:
        cmd_args.extend(["--model_path_2d", args.model_path_2d])
    if args.save_imgs:
        cmd_args.append("--save_imgs")
        
    subprocess.run(cmd_args, check=True)
    logwriter.write("2D GaussianImage training completed")

    # Combine the layers
    gaussianimage_rendered_path = Path(f"./checkpoints/{args.data_name}/{args.model_name_2d}_{args.iterations_2d}_{args.num_points_2d}")
    gaussianimage_rendered_images = glob.glob(os.path.join(gaussianimage_rendered_path, '*', f"frame_*_fitting.png"))
    gaussianimage_rendered_images.sort(key=lambda x: int(x.split('_')[-2]))  # Sort by frame number
    final_rendered_path = os.path.join(final_dir_path, "final_rendered")
    # os.makedirs(final_rendered_path, exist_ok=True)
    # for i in range(len(gaussianimage_rendered_images)): # copy rendered image to final_rendered_path
    #     shutil.copy(gaussianimage_rendered_images[i], os.path.join(final_rendered_path, os.path.basename(gaussianimage_rendered_images[i])))
    final_rendered_path = combine_layers(gaussianvideo_rendered_images, gaussianimage_rendered_images, final_dir_path)

    # Compare the final rendered images with the ground truth
    avg_psnr, avg_ms_ssim = evaluate_images(images_paths, final_rendered_path)
    logwriter.write("Final PSNR:{:.4f}, Final MS-SSIM:{:.4f}".format(avg_psnr, avg_ms_ssim))

    # move the folder into the final directory
    if os.path.exists(os.path.join(final_dir_path, os.path.basename(gaussianimage_rendered_path))):
        logwriter.write(f"Folder {os.path.basename(gaussianimage_rendered_path)} already exists in {final_dir_path}, removing it.")
        shutil.rmtree(os.path.join(final_dir_path, os.path.basename(gaussianimage_rendered_path)))
    shutil.copytree(os.path.dirname(gaussianvideo_rendered_path), os.path.join(final_dir_path, str(gaussianvideo_rendered_path).split('/')[-2]), dirs_exist_ok=True)
    shutil.move(gaussianimage_rendered_path, final_dir_path)
    logwriter.write(f"Moved GaussianVideo and GaussianImage folders to {final_dir_path}")

if __name__ == "__main__":
    main(sys.argv[1:])
