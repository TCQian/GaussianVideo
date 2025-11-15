import math
import os
from re import A
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

from utils import image_path_to_tensor, images_paths_to_tensor
from gaussianvideo3D2D import GaussianVideo3D2D
from gaussianimage_cholesky import GaussianImage_Cholesky

class GaussianVideo3D2DTrainerQuantize:
    """Tests quantized GaussianVideo models on a video (i.e. a set of frames)."""
    def __init__(
        self,
        images_paths: list[Path],
        num_points: int = 2000,
        model_name: str = "GaussianVideo",
        iterations: int = 30000,
        model_path = None,
        args = None,
        video_name: str = "Video",
        num_frames: int = 50,
        start_frame: int = 0,
        log_dir: Path = None,
    ):

        self.gt_video = images_paths_to_tensor(images_paths).to(self.device)  # [1, C, H, W, T]
        self.num_points = num_points
        self.video_name = video_name

        # Determine spatial and temporal dimensions
        self.H, self.W, self.T = self.gt_video.shape[2], self.gt_video.shape[3], self.gt_video.shape[4]
        BLOCK_H, BLOCK_W, BLOCK_T = 16, 16, 1  # adjust BLOCK_T if needed
        self.iterations = iterations
        self.iterations = args.iterations
        self.num_points = args.num_points
        self.save_imgs = args.save_imgs
        self.log_dir = log_dir

        if self.model_name == "GV3D2D":
            self.gaussian_model = GaussianVideo3D2D(
                layer=self.layer,
                loss_type="L2", 
                opt_type="adan", 
                H=self.H, 
                W=self.W, 
                T=self.T, 
                BLOCK_H=BLOCK_H, 
                BLOCK_W=BLOCK_W, 
                BLOCK_T=BLOCK_T, 
                device=self.device, 
                quantize=True,
                num_points=self.num_points,
                iterations=self.iterations,
                lr=args.lr
            ).to(self.device)

            self.gaussian_model._create_data_from_checkpoint(args.model_path_layer0, args.model_path_layer1)

        elif self.model_name == "GVGI":
            assert self.layer == 1, "GVGI is only able to process Layer 1 "
            checkpoint_layer0 = torch.load(args.model_path_layer0, map_location=self.device)
            self.init_num_points_layer1 = int((self.num_points * self.T) - checkpoint_layer0['_xyz_3D'].shape[0])
            num_points_per_frame = int(self.init_num_points_layer1 / self.T)
            print(f"GVGI: Available number of gaussians: {self.init_num_points_layer1} for layer 1")

            self.gaussian_model_list = []
            for t in range(self.T):
                if args.model_path_layer1 is not None:
                    checkpoint_file_path = Path(args.model_path_layer1) / f'frame_{t+1:04}' / f"gaussian_model.pth.tar"
                    if checkpoint_file_path.exists():
                        checkpoint = torch.load(checkpoint_file_path, map_location=self.device)
                        num_points = checkpoint['_xyz'].shape[0]
                else:                
                    if t == self.T - 1:
                        num_points = self.init_num_points_layer1 - (t * num_points_per_frame)
                    else:
                        num_points = num_points_per_frame

                background_path = Path(f"{args.model_path_layer0.replace('layer_0_model.pth.tar', '')}/{self.video_name}_fitting_t{t}_layer0.png")
                background_img = image_path_to_tensor(background_path).squeeze(0).permute(1, 2, 0).to(self.device)
                gaussian_model = GaussianImage_Cholesky(
                    background_image=background_img,
                    loss_type="L2", 
                    opt_type="adan", 
                    num_points=num_points,
                    H=self.H, 
                    W=self.W, 
                    BLOCK_H=BLOCK_H, 
                    BLOCK_W=BLOCK_W, 
                    device=self.device, 
                    quantize=True,
                    iterations=self.iterations,
                    lr=args.lr
                ).to(self.device)

                if args.model_path_layer1:
                    gaussian_model.load_state_dict(checkpoint)
                    print(f"Loaded checkpoint from: {checkpoint_file_path}")

                self.gaussian_model_list.append(gaussian_model)

        self.logwriter = LogWriter(self.log_dir, train=False)

        if model_path is not None:
            full_model_path = os.path.join(model_path, args.data_name, "gaussian_model.best.pth.tar")
            print(f"loading model path: {full_model_path}")
            checkpoint = torch.load(full_model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def test_GV3D2D(self, gaussian_model, gt_image):
        gaussian_model.eval()
        with torch.no_grad():
            encoding_dict = gaussian_model.compress_wo_ec()
            out = gaussian_model.decompress_wo_ec(encoding_dict)
            start_time = time.time()
            for i in range(100):
                _ = gaussian_model.decompress_wo_ec(encoding_dict)
            end_time = (time.time() - start_time) / 100
        data_dict = gaussian_model.analysis_wo_ec(encoding_dict)

        out_video = out["render"].float()  # Expected shape: [1, C, H, W, T]
        mse_loss = F.mse_loss(out_video, gt_image)
        psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-8))
        
        T = out_video.shape[-1]
        ms_ssim_total = 0.0
        for t in range(T):
            frame_pred = out_video[..., t]  # shape: [1, C, H, W]
            frame_gt = gt_image[..., t]  # shape: [1, C, H, W]
            ms_ssim_total += ms_ssim(frame_pred, frame_gt, data_range=1, size_average=True).item()
        ms_ssim_value = ms_ssim_total / T

        data_dict["psnr"] = psnr
        data_dict["ms-ssim"] = ms_ssim_value
        data_dict["rendering_time"] = end_time
        data_dict["rendering_fps"] = 1 / end_time

        np.save(self.log_dir / f"test_layer_{self.layer}.npy", data_dict)
        self.logwriter.write("Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, 1 / end_time))
        self.logwriter.write("PSNR:{:.4f}, MS_SSIM:{:.6f}, bpp:{:.4f}".format(psnr, ms_ssim_value, data_dict["bpp"]))
        self.logwriter.write("position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}".format(
            data_dict["position_bpp"], data_dict["cholesky_bpp"], data_dict["feature_dc_bpp"]))
        return data_dict

    def test_GVGI(self, gaussian_model, gt_image, t=0):
        gaussian_model.eval()
        with torch.no_grad():
            encoding_dict = gaussian_model.compress_wo_ec()
            out = gaussian_model.decompress_wo_ec(encoding_dict)
            start_time = time.time()
            for i in range(100):
                _ = gaussian_model.decompress_wo_ec(encoding_dict)
            end_time = (time.time() - start_time)/100
        data_dict = gaussian_model.analysis_wo_ec(encoding_dict)
    
        out_img = out["render"].float()
        mse_loss = F.mse_loss(out_img, gt_image)
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out_img, gt_image, data_range=1, size_average=True).item()
        
        data_dict["psnr"] = psnr
        data_dict["ms-ssim"] = ms_ssim_value
        data_dict["rendering_time"] = end_time
        data_dict["rendering_fps"] = 1/end_time
        np.save(self.log_dir / f"test_layer_{self.layer}.npy", data_dict)
        self.logwriter.write("Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, 1/end_time))
        self.logwriter.write("PSNR:{:.4f}, MS_SSIM:{:.6f}, bpp:{:.4f}".format(psnr, ms_ssim_value, data_dict["bpp"]))
        self.logwriter.write("position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}".format(data_dict["position_bpp"], data_dict["cholesky_bpp"], data_dict["feature_dc_bpp"]))
        return data_dict

    def test(self):
        if self.model_name == "GV3D2D":
            data_dict = self.test_GV3D2D(self.gaussian_model, self.gt_video)
        elif self.model_name == "GVGI":
            psnr_list = []
            ms_ssim_list = []
            rendering_time_list = []
            rendering_fps_list = []
            position_bpp_list = []
            cholesky_bpp_list = []
            feature_dc_bpp_list = []
            for t in range(self.T):
                data_dict = self.test_GVGI(self.gaussian_model_list[t], self.gt_video[..., t], t)
                psnr_list.append(data_dict["psnr"])
                ms_ssim_list.append(data_dict["ms-ssim"])
                rendering_time_list.append(data_dict["rendering_time"])
                rendering_fps_list.append(data_dict["rendering_fps"])
                position_bpp_list.append(data_dict["position_bpp"])
                cholesky_bpp_list.append(data_dict["cholesky_bpp"])
                feature_dc_bpp_list.append(data_dict["feature_dc_bpp"])

            data_dict = {}
            data_dict["psnr"] = sum(psnr_list) / len(psnr_list)
            data_dict["ms-ssim"] = sum(ms_ssim_list) / len(ms_ssim_list)
            data_dict["rendering_time"] = sum(rendering_time_list) / len(rendering_time_list)
            data_dict["rendering_fps"] = sum(rendering_fps_list) / len(rendering_fps_list)
            data_dict["position_bpp"] = sum(position_bpp_list) / len(position_bpp_list)
            data_dict["cholesky_bpp"] = sum(cholesky_bpp_list) / len(cholesky_bpp_list)
            data_dict["feature_dc_bpp"] = sum(feature_dc_bpp_list) / len(feature_dc_bpp_list)
        
        return data_dict

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Test quantized GaussianVideo3D2D model.")
    
    # Parameters for training
    parser.add_argument("--seed", type=int, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")

    # Progressively training parameters
    parser.add_argument("--layer", type=int, default=0, help="Target layer to train (default: %(default)s)")
    
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

    # Model parameters for layer 0
    parser.add_argument(
        "--model_name", type=str, default="GV3D2D", help="model selection: GaussianVideo3D2D"
    )

    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="3D GS points (default: %(default)s)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate of layer 0 (default: %(default)s)",
    )
    parser.add_argument("--model_path_layer0", type=str, default=None, help="Path to a layer 0 checkpoint")

    parser.add_argument("--model_path_layer1", type=str, default=None, help="Path to a layer 1 checkpoint")
    
    parser.add_argument("--quantize", action="store_true", help="Enable quantization")

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

    log_dir = Path(f"./checkpoints_quant/{args.data_name}/ProgressiveGaussianVideo_i{args.iterations}_g{args.num_points}_f{args.num_frames}_s{args.start_frame}/layer{args.layer}/")
    if args.layer == 1:
        log_dir = log_dir / (f"{args.model_name}_i{args.iterations}_g{args.num_points}/")
    
    logwriter = LogWriter(log_dir)

    images_paths = []
    for i in range(args.start_frame, args.start_frame + args.num_frames):
        image_path = Path(args.dataset) / f'frame_{i+1:04}.png'
        images_paths.append(image_path)

    trainer = GaussianVideo3D2DTrainerQuantize(layer=args.layer, images_paths=images_paths, model_name=args.model_name, args=args, num_frames=args.num_frames, start_frame=args.start_frame, video_name=args.data_name, model_path=args.model_path)
    
    data_dict = trainer.test()
    logwriter.write("Video: {}x{}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, bpp:{:.4f}, Eval time:{:.8f}s, FPS:{:.4f}, position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}".format(
        trainer.H, trainer.W, trainer.T, data_dict["psnr"], data_dict["ms-ssim"], data_dict["bpp"],
        data_dict["rendering_time"], data_dict["rendering_fps"],
        data_dict["position_bpp"], data_dict["cholesky_bpp"], data_dict["feature_dc_bpp"]
    ))
    
if __name__ == "__main__":
    main(sys.argv[1:])
