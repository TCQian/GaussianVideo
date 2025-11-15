import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tqdm import tqdm
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
    Loads a list of image paths and stacks them into a 5D tensor of shape [1, C, H, W, T].
    """
    image_tensors = []
    for image_path in images_paths:
        img_tensor = image_path_to_tensor(image_path)
        image_tensors.append(img_tensor)
    # Stack along a new time dimension; each tensor is [1, C, H, W] so stacking gives [T, 1, C, H, W]
    stacked = torch.stack(image_tensors, dim=0)
    # Permute to get [1, C, H, W, T]
    final_tensor = stacked.permute(1, 2, 3, 4, 0)
    return final_tensor

class SimpleTrainerVideoQuantize:
    """Trains 2D+t gaussians (video) with quantization to fit a video."""
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
    ):
        self.device = torch.device("cuda:0")
        self.gt_video = images_paths_to_tensor(images_paths).to(self.device)  # [1, C, H, W, T]
        self.num_points = num_points
        self.video_name = video_name

        # Determine spatial and temporal dimensions
        self.H, self.W, self.T = self.gt_video.shape[2], self.gt_video.shape[3], self.gt_video.shape[4]
        BLOCK_H, BLOCK_W, BLOCK_T = 16, 16, 1  # adjust BLOCK_T if needed
        self.iterations = iterations
        # Incorporate num_frames and start_frame into the log directory name
        self.log_dir = Path(
            f"./checkpoints_quant/{args.data_name}/{model_name}_i{args.iterations_3d}_g{num_points}_f{num_frames}_s{start_frame}/{video_name}"
        )
        self.save_imgs = args.save_imgs

        # Instantiate the video model (with quantization enabled)
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

        self.logwriter = LogWriter(self.log_dir)
        if model_path is not None:
            full_model_path = os.path.join(model_path, args.data_name, "gaussian_model.pth.tar")
            print(f"loading model path:{full_model_path}")
            checkpoint = torch.load(full_model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)
            self.gaussian_model._init_data()

    def train(self):     
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        best_psnr = 0
        self.gaussian_model.train()
        start_time = time.time()
        best_model_dict = None
        for iter in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.train_iter_quantize(self.gt_video)
            psnr_list.append(psnr)
            iter_list.append(iter)
            if psnr > best_psnr:
                best_psnr = psnr
                best_model_dict = copy.deepcopy(self.gaussian_model.state_dict())
            if iter % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{loss.item():.7f}",
                    "PSNR": f"{psnr:.4f}",
                    "Best PSNR": f"{best_psnr:.4f}"
                })
                progress_bar.update(10)
        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value, bpp = self.test(best=False)
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        self.gaussian_model.load_state_dict(best_model_dict)
        best_psnr_value, best_ms_ssim_value, best_bpp = self.test(best=True)
        torch.save(best_model_dict, self.log_dir / "gaussian_model.best.pth.tar")
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model.forward_quantize()
            test_end_time = (time.time() - test_start_time) / 100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}"
                             .format(end_time, test_end_time, 1/test_end_time))
        np.save(self.log_dir / "training.npy", {
            "iterations": iter_list,
            "training_psnr": psnr_list,
            "training_time": end_time,
            "psnr": psnr_value,
            "ms-ssim": ms_ssim_value,
            "rendering_time": test_end_time,
            "rendering_fps": 1/test_end_time,
            "bpp": bpp,
            "best_psnr": best_psnr_value,
            "best_ms-ssim": best_ms_ssim_value,
            "best_bpp": best_bpp
        })
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time, bpp, best_psnr_value, best_ms_ssim_value, best_bpp

    def test(self, best=False):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model.forward_quantize()
        out_video = out["render"].float()  # Expected shape: [1, C, H, W, T]
        self.gt_video = self.gt_video.float()
        mse_loss = F.mse_loss(out_video, self.gt_video)
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        
        # Compute MS-SSIM frame-by-frame.
        num_time_steps = out_video.size(-1)  # T dimension
        ms_ssim_total = 0.0
        ms_ssim_values = []
        try:
            for t in range(num_time_steps):
                # Extract the t-th frame: shape becomes [1, C, H, W]
                frame = out_video[:, :, :, :, t]
                gt_frame = self.gt_video[:, :, :, :, t]
                ms_ssim_values.append(
                    ms_ssim(frame, gt_frame, data_range=1, size_average=True).item()
                )
            ms_ssim_value = sum(ms_ssim_values) / len(ms_ssim_values)
        except AssertionError as e:
            self.logwriter.write("MS-SSIM could not be computed: " + str(e))
            ms_ssim_value = 0.0

        m_bit, s_bit, r_bit, c_bit = out["unit_bit"]
        # Compute bits per pixel per frame (using H and W).
        bpp = (m_bit + s_bit + r_bit + c_bit) / (self.H * self.W * self.T)
        tag = "Best Test" if best else "Test"
        self.logwriter.write(f"{tag} PSNR:{psnr:.4f}, MS_SSIM:{ms_ssim_value:.6f}, bpp:{bpp:.4f}")

        if self.save_imgs:
            transform = transforms.ToPILImage()
    
            # Loop through each time step
            for t in range(num_time_steps):
                # Extract the image for the current time step
                img = out_video[0, :, :, :, t]  # Shape: [C, H, W]
                
                # Convert the image tensor to a PIL Image
                pil_image = transform(img)  # Convert to PIL Image
                
                # Define the filename based on the time step
                name = f"{self.video_name}_fitting_t{t}.png"  # e.g., "_fitting_t0.png"
                
                # Save the image to the specified directory
                pil_image.save(str(self.log_dir / name))
        return psnr, ms_ssim_value, bpp

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train quantized GaussianVideo model.")
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
    parser.add_argument("--save_imgs", default=True, action="store_true", help="Save rendered frames")
    parser.add_argument("--lr_3d", type=float, default=1e-2, help="Learning rate of 3D GaussianVideo (default: %(default)s)")
    parser.add_argument("--pretrained", type=str, help="Path to a checkpoint")
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
    
    # Build the list of frame image paths based on start_frame and num_frames.
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
        video_name=args.data_name,
        num_frames=args.num_frames,
        start_frame=args.start_frame,
        model_path=args.model_path_3d
    )
    psnr, ms_ssim, training_time, eval_time, eval_fps, bpp, best_psnr, best_ms_ssim, best_bpp = trainer.train()
    logwriter.write("Final Results - PSNR:{:.4f}, MS-SSIM:{:.4f}, bpp:{:.4f}, Best PSNR:{:.4f}, Best MS-SSIM:{:.4f}, Best bpp:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
            psnr, ms_ssim, bpp, best_psnr, best_ms_ssim, best_bpp, training_time, eval_time, eval_fps))

if __name__ == "__main__":
    main(sys.argv[1:])
