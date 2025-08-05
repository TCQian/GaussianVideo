import math
import os
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
from utils import *
from tqdm import tqdm
from collections import OrderedDict
import random
import copy
import torchvision.transforms as transforms

def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return img_tensor

def images_paths_to_tensor(images_paths: list[Path]):
    """
    Loads a list of image paths and stacks them into a 5D tensor of shape [1, C, H, W, T]
    """
    image_tensors = []
    for image_path in images_paths:
        img_tensor = image_path_to_tensor(image_path)
        image_tensors.append(img_tensor)
    # Stack along a new time dimension; images are [1, C, H, W] so stacking gives [T, 1, C, H, W]
    stacked = torch.stack(image_tensors, dim=0)
    # Permute to get [1, C, H, W, T]
    final_tensor = stacked.permute(1, 2, 3, 4, 0)
    return final_tensor

class SimpleTrainerVideoQuantize:
    """Tests quantized GaussianVideo models on a video (i.e. a set of frames)."""
    def __init__(
        self,
        images_paths: list[Path],
        num_points: int = 2000,
        model_name: str = "GaussianVideo",
        iterations: int = 30000,
        model_path = None,
        args = None,
    ):
        self.device = torch.device("cuda:0")
        self.gt_video = images_paths_to_tensor(images_paths).to(self.device)  # [1, C, H, W, T]
        self.num_points = num_points

        # Determine dimensions from the ground-truth video tensor.
        self.H, self.W, self.T = self.gt_video.shape[2], self.gt_video.shape[3], self.gt_video.shape[4]
        BLOCK_H, BLOCK_W, BLOCK_T = 16, 16, 1
        self.iterations = iterations
        image_name = images_paths[0].stem  # using first frame's name as an identifier

        # Incorporate num_frames and start_frame into the log directory name.
        self.log_dir = Path(
            f"./checkpoints_quant/{args.data_name}/{model_name}_i{args.iterations_3d}_g{num_points}_f{args.num_frames}_s{args.start_frame}/{image_name}"
        )

        from gaussianvideo import GaussianVideo
        self.gaussian_model = GaussianVideo(
            loss_type="L2",
            opt_type="adan",
            num_points=self.num_points,
            H=self.H,
            W=self.W,
            T=self.T,
            BLOCK_H=BLOCK_H,
            BLOCK_W=BLOCK_W,
            BLOCK_T=BLOCK_T,
            device=self.device,
            lr=args.lr_3d,
            quantize=True
        ).to(self.device)
        
        self.logwriter = LogWriter(self.log_dir, train=False)

        if model_path is not None:
            print(f"loading model path: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            # Use the quantization pipeline (without entropy coding)
            encoding_dict = self.gaussian_model.compress_wo_ec()
            out = self.gaussian_model.decompress_wo_ec(encoding_dict)
            start_time = time.time()
            # Time 100 decompressions
            for i in range(100):
                _ = self.gaussian_model.decompress_wo_ec(encoding_dict)
            end_time = (time.time() - start_time) / 100
        data_dict = self.gaussian_model.analysis_wo_ec(encoding_dict)

        out_video = out["render"].float()  # Expected shape: [1, C, H, W, T]
        mse_loss = F.mse_loss(out_video, self.gt_video)
        psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-8))
        
        # Compute MS-SSIM by averaging over frames.
        T = out_video.shape[-1]
        ms_ssim_total = 0.0
        for t in range(T):
            frame_pred = out_video[..., t]  # shape: [1, C, H, W]
            frame_gt = self.gt_video[..., t]  # shape: [1, C, H, W]
            ms_ssim_total += ms_ssim(frame_pred, frame_gt, data_range=1, size_average=True).item()
        ms_ssim_value = ms_ssim_total / T

        data_dict["psnr"] = psnr
        data_dict["ms-ssim"] = ms_ssim_value
        data_dict["rendering_time"] = end_time
        data_dict["rendering_fps"] = 1 / end_time

        np.save(self.log_dir / "test.npy", data_dict)
        self.logwriter.write("Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, 1 / end_time))
        self.logwriter.write("PSNR:{:.4f}, MS_SSIM:{:.6f}, bpp:{:.4f}".format(psnr, ms_ssim_value, data_dict["bpp"]))
        self.logwriter.write("position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}".format(
            data_dict["position_bpp"], data_dict["cholesky_bpp"], data_dict["feature_dc_bpp"]))
        return data_dict

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Test quantized GaussianVideo model.")
    parser.add_argument("-d", "--dataset", type=str, default='./dataset/Jockey/', help="Dataset directory containing video frames")
    parser.add_argument("--data_name", type=str, default='Jockey', help="Dataset name")
    parser.add_argument("--iterations_3d", type=int, default=50000, help="number of training epochs for 3D GaussianVideo (default: %(default)s)")
    parser.add_argument("--model_name_3d", type=str, default="GaussianVideo", help="model selection for 3D: GaussianVideo")
    parser.add_argument("--num_points_3d", type=int, default=50000, help="2D+T GS points (default: %(default)s)")
    parser.add_argument("--num_frames", type=int, default=50, help="Number of frames to use")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--model_path_3d", type=str, default=None, help="Path to a 3D GaussianVideo's checkpoint")
    parser.add_argument("--seed", type=int, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--quantize", action="store_true", help="Enable quantization")
    parser.add_argument("--save_imgs", action="store_true", help="Save rendered frames")
    parser.add_argument("--lr_3d", type=float, default=1e-2, help="Learning rate of 3D GaussianVideo (default: %(default)s)")
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    logwriter = LogWriter(Path(f"./checkpoints_quant/{args.data_name}/{args.model_name_3d}_i{args.iterations_3d}_g{args.num_points_3d}_f{args.num_frames}_s{args.start_frame}"), train=False)
    
    # Build the list of image paths for the video frames.
    images_paths = []
    for i in range(args.start_frame, args.start_frame + args.num_frames):
        image_path = Path(args.dataset) / f'frame_{i+1:04}.png'
        images_paths.append(image_path)

    trainer = SimpleTrainerVideoQuantize(
        images_paths=images_paths,
        num_points=args.num_points_3d,
        iterations=args.iterations_3d,
        model_name=args.model_name_3d,
        args=args,
        model_path=args.model_path_3d
    )
    
    data_dict = trainer.test()
    logwriter.write("Video: {}x{}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, bpp:{:.4f}, Eval time:{:.8f}s, FPS:{:.4f}, position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}".format(
        trainer.H, trainer.W, trainer.T, data_dict["psnr"], data_dict["ms-ssim"], data_dict["bpp"],
        data_dict["rendering_time"], data_dict["rendering_fps"],
        data_dict["position_bpp"], data_dict["cholesky_bpp"], data_dict["feature_dc_bpp"]
    ))
    
if __name__ == "__main__":
    main(sys.argv[1:])
