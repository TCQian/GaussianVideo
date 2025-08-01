from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
import random
import glob

from utils import *
from train_video import VideoTrainer
from train import SimpleTrainer2d

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
   
    # Parameters for training
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
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
    psnrs, ms_ssims, training_times, eval_times, eval_fpses = [], [], [], [], []
    image_h, image_w = 0, 0
    image_length, start = args.num_frames, args.start_frame

    images_paths = []
    for i in range(start, start+image_length):
        image_path = Path(args.dataset) / f'frame_{i+1:04}.png'
        images_paths.append(image_path)
        
    trainer = VideoTrainer(images_paths=images_paths, num_points=args.num_points_3d,
        iterations=args.iterations_3d, model_name=args.model_name_3d, args=args, model_path=args.model_path_3d, num_frames=args.num_frames, start_frame=args.start_frame, video_name=args.data_name)
    
    psnr, ms_ssim, training_time, eval_time, eval_fps = trainer.train()
    psnrs.append(psnr)
    ms_ssims.append(ms_ssim)
    training_times.append(training_time) 
    eval_times.append(eval_time)
    eval_fpses.append(eval_fps)
    image_h += trainer.H
    image_w += trainer.W
    image_name = image_path.stem
    logwriter.write("{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training GaussianVideo as Layer 1:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        image_name, trainer.H, trainer.W, psnr, ms_ssim, training_time, eval_time, eval_fps))

    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    # avg_h = image_h//image_length
    # avg_w = image_w//image_length

    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training GaussianVideo as Layer 1:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        image_h, image_w, avg_psnr, avg_ms_ssim, avg_training_time, avg_eval_time, avg_eval_fps))    
    
    # Collect delta image for 2D GaussianImage training
    gaussianvideo_rendered_path = Path(f"./checkpoints/{args.data_name}/{args.model_name_3d}_i{args.iterations_3d}_g{args.num_points_3d}_f{args.num_frames}_s{args.start_frame}/{args.data_name}")
    gaussianvideo_rendered_images = glob.glob(os.path.join(gaussianvideo_rendered_path, f"{args.data_name}_fitting_t*.png"))
    gaussianvideo_rendered_images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0][1:]))  # Sort by frame number
    assert gaussianvideo_rendered_path.exists(), "GaussianVideo rendered images not found. Please check the training of 3D GaussianVideo."
    assert len(gaussianvideo_rendered_images) == image_length, "Number of rendered images does not match the number of frames."

    output_delta_path = os.path.join(final_dir_path, "delta_images")
    os.makedirs(output_delta_path, exist_ok=True)

    delta_images_paths = []
    for gt_img_path, rendered_img_path in zip(images_paths, gaussianvideo_rendered_images):
        print(f"Processing {gt_img_path} and {rendered_img_path}")
        gt_image = cv2.imread(str(gt_img_path), cv2.IMREAD_UNCHANGED)
        rendered_image = cv2.imread(rendered_img_path, cv2.IMREAD_UNCHANGED)
        delta_image = cv2.absdiff(gt_image, rendered_image)
        # Save the delta image
        delta_image_path = os.path.join(output_delta_path, os.path.basename(gt_img_path))
        cv2.imwrite(delta_image_path, delta_image)
        delta_images_paths.append(delta_image_path)

    # Training 2D GaussianImage as Layer 2
    psnrs, ms_ssims, training_times, eval_times, eval_fpses = [], [], [], [], []
    image_h, image_w = 0, 0

    for img_path in delta_images_paths:
        trainer = SimpleTrainer2d(image_path=img_path, num_points=args.num_points_2d, 
            iterations=args.iterations_2d, model_name=args.model_name_2d, args=args, model_path=args.model_path_2d)
        psnr, ms_ssim, training_time, eval_time, eval_fps = trainer.train()
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        training_times.append(training_time) 
        eval_times.append(eval_time)
        eval_fpses.append(eval_fps)
        image_h += trainer.H
        image_w += trainer.W
        image_name = image_path.stem
        logwriter.write("{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training for 2D GaussianImage:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
            image_name, trainer.H, trainer.W, psnr, ms_ssim, training_time, eval_time, eval_fps))

    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_h = image_h//image_length
    avg_w = image_w//image_length

    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training for 2D GaussianImage:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_training_time, avg_eval_time, avg_eval_fps))

if __name__ == "__main__":
    main(sys.argv[1:])
