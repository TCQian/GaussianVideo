from gsplat.project_gaussians_video import project_gaussians_video
from gsplat.rasterize_sum_video import rasterize_gaussians_sum_video
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from optimizer import Adan
from PIL import Image
import torchvision.transforms as transforms
import sys
import time
from pathlib import Path
import argparse
import yaml
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from tqdm import tqdm
import random
class GaussianVideo_Layer(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()

        self.debug_mode = False  # enable kernel logging

        self.layer = 0
        self.loss_type = loss_type
        # self.init_num_points_3D = 2500
        self.init_num_points_2D = kwargs["num_points"]

        self.H, self.W, self.T = kwargs["H"], kwargs["W"], kwargs["T"]
        self.BLOCK_W, self.BLOCK_H, self.BLOCK_T = kwargs["BLOCK_W"], kwargs["BLOCK_H"], kwargs["BLOCK_T"]
        
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W, # (1920 + 16 - 1) // 16 = 120
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H, # (1080 + 16 - 1) // 16 = 68
            (self.T + self.BLOCK_T - 1) // self.BLOCK_T, # (50 + 1 - 1) // 1 = 50
        ) # tile_bounds (120, 68, 50)
        self.device = kwargs["device"]

        self.register_buffer('background', torch.ones(3))
        self.register_buffer('cholesky_bound_3D', torch.tensor([0.5, 0, 0.5, 0.5, 0, 0.5]).view(1, 6))
        self.register_buffer('cholesky_bound_2D', torch.tensor([0.5, 0, 0, 0.5, 0, 0]).view(1, 3))

        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        
        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    def _init_layer0(self):
        self._xyz_3D = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points_3D, 3) - 0.5)))
        self._cholesky_3D = nn.Parameter(torch.rand(self.init_num_points_3D, 6))
        self.register_buffer('_opacity_3D', torch.ones((self.init_num_points_3D, 1)))
        self._features_dc_3D = nn.Parameter(torch.rand(self.init_num_points_3D, 3))
        
        # Increase L33 (the last element in each row) to boost temporal variance.
        with torch.no_grad():
            self._cholesky_3D.data[:, 5] += self.T  # adjust the constant as needed

        self.layer = 0

    def _init_layer1(self):
        self._opacity_3D = nn.Parameter(torch.logit(0.1 * torch.ones(self.init_num_points_3D, 1)))

        # 2D -> 3D, xy is random, z is specified frame number from 0 to T-1
        self._xyz_2D = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points_2D * self.T, 3) - 0.5)))
        for t in range(self.T):
            self._xyz_2D[t * self.init_num_points_2D:(t + 1) * self.init_num_points_2D, 2] = torch.atanh((2 * t) - 0.5)

        # 2D -> 3D, L11, L12, L22 are random, L13, L23 are 0 and L33 is 1
        self._cholesky_2D = nn.Parameter(torch.rand(self.init_num_points_2D, 6))
        with torch.no_grad():
            self._cholesky_2D.data[:, 2] = 0
            self._cholesky_2D.data[:, 4] = 0
            self._cholesky_2D.data[:, 5] = 1

        # self._opacity_2D = nn.Parameter(torch.logit(0.1 * torch.ones(self.init_num_points_2D, 1)))
        self.register_buffer('_opacity_2D', torch.ones((self.init_num_points_2D, 1)))
        self._features_dc_2D = nn.Parameter(torch.rand(self.init_num_points_2D, 3))
        self.layer = 1

    def merge_3D2D(self):
        self._merged_xyz = torch.nn.Parameter(torch.cat((self._xyz_3D, self._xyz_2D), dim=0))
        self._merged_cholesky = torch.nn.Parameter(torch.cat((self._cholesky_3D, self._cholesky_2D), dim=0))
        self._merged_features_dc = torch.nn.Parameter(torch.cat((self._features_dc_3D, self._features_dc_2D), dim=0))
        self._merged_opacity = torch.nn.Parameter(torch.cat((self._opacity_3D, self._opacity_2D), dim=0))

    def freeze_gaussian(self):
        if self.layer == 1:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    if name == "_xyz_2D":
                        param.grad[:, 2] = 0
                        param[:, 2].requires_grad_(False)
                    elif name == "_cholesky_2D":
                        param.grad[:, [2, 4, 5]] = 0
                        param[:, [2, 4, 5]].requires_grad_(False)

    @property
    def get_xyz(self):
        if self.layer == 0:
            return torch.tanh(self._xyz_3D)
        elif self.layer == 1:
            return torch.tanh(self._xyz_2D)
    
    @property
    def get_features(self):
        if self.layer == 0:
            return self._features_dc_3D
        elif self.layer == 1:
            return self._features_dc_2D
    
    @property
    def get_opacity(self):
        if self.layer == 0:
            return self._opacity_3D
        elif self.layer == 1:
            return self._opacity_2D
            # return self.opacity_activation(self._opacity_2D)
    
    @property
    def get_cholesky_elements(self):
        if self.layer == 0:
            return self._cholesky_3D + self.cholesky_bound_3D
        elif self.layer == 1:
            return self._cholesky_2D + self.cholesky_bound_2D
    
    def forward(self):
        # print("before projection, xyz: {xyz}, cholesky: {cholesky}".format(xyz=self.get_xyz, cholesky=self.get_cholesky_elements))
        self.xys, depths, radii, conics, num_tiles_hit = project_gaussians_video(
            self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.T, self.tile_bounds
        )
        out_img = rasterize_gaussians_sum_video(
            self.xys, depths, radii, conics, num_tiles_hit,
            self.get_features, self.get_opacity, self.H, self.W, self.T,
            self.BLOCK_H, self.BLOCK_W, self.BLOCK_T,
            background=self.background, return_alpha=False
        )
        # if self.debug_mode:
            # radii_np = self.radii.detach().cpu().numpy()
            # max_radius = np.ceil(radii_np.max() / 5) * 5
            # bins = np.arange(0, max_radius + 5, 5)  # e.g., [0, 5, 10, ..., max]

            # hist, bin_edges = np.histogram(radii_np, bins=bins)

            # # Print histogram nicely
            # print("Gaussian Radius Histogram (bin size = 5)")
            # for i in range(len(hist)):
            #     print(f"[{bin_edges[i]:>2.0f} - {bin_edges[i+1]:>2.0f}) : {hist[i]} Gaussians")
            # self.debug_mode = False
        out_img = torch.clamp(out_img, 0, 1)  # [T, H, W, 3]
        out_img = out_img.view(-1, self.T, self.H, self.W, 3).permute(0, 4, 2, 3, 1).contiguous()
        return {"render": out_img}


    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()

        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-8))

        if self.layer == 1:
            self.freeze_gaussian()

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        
        return loss, psnr


class VideoTrainer_Layer:
    """Trains 3D and 2D gaussians layer by layer to fit a video."""
    def __init__(
        self,
        images_paths: list[Path],
        num_points: int = 2000,
        model_name:str = "GaussianVideo_Layer",
        iterations:int = 30000,
        model_path = None,
        args = None,
        video_name: str = "Jockey",
        num_frames: int = 50,
        start_frame: int = 0,
    ):
        self.video_name = video_name
        self.device = torch.device("cuda:0")
        self.gt_image = images_paths_to_tensor(images_paths).to(self.device)

        self.num_points = num_points
        BLOCK_H, BLOCK_W, BLOCK_T = 16, 16, 1
        self.H, self.W, self.T = self.gt_image.shape[2], self.gt_image.shape[3], self.gt_image.shape[4]
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.log_dir = Path(f"./checkpoints/{args.data_name}/{model_name}_i{args.iterations_3d}_g{num_points}_f{num_frames}_s{start_frame}/{video_name}")
            
        if model_name == "GaussianVideo_Layer":
            self.gaussian_model = GaussianVideo_Layer(
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
                quantize=False
            ).to(self.device)

        self.logwriter = LogWriter(self.log_dir)

        if model_path is not None:
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def train(self):     
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        best_psnr = 0
        self.gaussian_model.train()
        start_time = time.time()
        for iter in range(1, self.iterations+1):
            if iter == 1 or iter % 1000 == 0:
                self.gaussian_model.debug_mode = True
            else:
                self.gaussian_model.debug_mode = False
            # if iter % 5000 == 1 and iter > 1:
            #     self.gaussian_model.prune(tile_threshold=2.0) # prune gaussians affecting <= 2 tiles

            loss, psnr = self.gaussian_model.train_iter(self.gt_image)
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value = self.test()
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        np.save(self.log_dir / "training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time})
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time

    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        
        # Assuming out["render"] is of shape [1, C, H, W, T]
        render_tensor = out["render"].float()  # Ensure the tensor is in float format

        # Get the number of time steps (T)
        num_time_steps = render_tensor.size(-1)  # T dimension
        
        ms_ssim_value = 0.0
        try:
            ms_ssim_values = []
            for t in range(num_time_steps):
                # Extract the t-th frame from both render and ground truth
                frame = render_tensor[..., t]  # e.g. shape: [1, 3, H, W]
                gt_frame = self.gt_image[..., t] # e.g. shape: [1, 3, H, W]
                # Attempt to compute MS-SSIM for this frame
                ms_ssim_values.append(
                    ms_ssim(frame, gt_frame, data_range=1, size_average=True).item()
                )
            ms_ssim_value = sum(ms_ssim_values) / len(ms_ssim_values)
        except AssertionError as e:
            # In case the image is too small for ms-ssim, log the error and continue.
            self.logwriter.write("MS-SSIM could not be computed: " + str(e))
        
        # Log the results based on whether MS-SSIM was computed
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
        
        if self.save_imgs:
            transform = transforms.ToPILImage()
    
            # Loop through each time step
            for t in range(num_time_steps):
                # Extract the image for the current time step
                img = render_tensor[0, :, :, :, t]  # Shape: [C, H, W]
                
                # Convert the image tensor to a PIL Image
                pil_image = transform(img)  # Convert to PIL Image
                
                # Define the filename based on the time step
                name = f"{self.video_name}_fitting_t{t}.png"  # e.g., "_fitting_t0.png"
                
                # Save the image to the specified directory
                pil_image.save(str(self.log_dir / name))
        return psnr, ms_ssim_value
    
def images_paths_to_tensor(images_paths: list[Path]):
    # Initialize a list to hold the image tensors
    image_tensors = []
    
    # Loop through each image path
    for image_path in images_paths:
        # Convert the image at the current path to a tensor
        img_tensor = image_path_to_tensor(image_path)
        # Append the tensor to the list
        image_tensors.append(img_tensor)
    
    # Stack the list of tensors along a new dimension to create a 5D tensor
    # This will result in a tensor of shape [T, 1, C, H, W]
    stacked_tensor = torch.stack(image_tensors, dim=0)  # Shape: [T, 1, C, H, W]
    
    # Rearrange the dimensions to get the final shape [1, C, H, W, T]
    final_tensor = stacked_tensor.permute(1, 2, 3, 4, 0)  # Shape: [1, C, H, W, T]
    
    return final_tensor

def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0) #[1, C, H, W]
    return img_tensor

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./dataset/Jockey/', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='Jockey', help="Training dataset"
    )
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
    parser.add_argument("--model_path_3d", type=str, default=None, help="Path to a 3D GaussianVideo's checkpoint")
    parser.add_argument("--seed", type=int, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image", default=True)
    parser.add_argument(
        "--lr_3d",
        type=float,
        default=1e-2,
        help="Learning rate of 3D GaussianVideo (default: %(default)s)",
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

    logwriter = LogWriter(Path(f"./checkpoints/{args.data_name}/{args.model_name_3d}_i{args.iterations_3d}_g{args.num_points_3d}_f{args.num_frames}_s{args.start_frame}"))
    psnrs, ms_ssims, training_times, eval_times, eval_fpses = [], [], [], [], []
    image_h, image_w = 0, 0
    image_length, start = args.num_frames, args.start_frame

    images_paths = []
    for i in range(start, start+image_length):
        image_path = Path(args.dataset) / f'frame_{i+1:04}.png'
        images_paths.append(image_path)
    trainer = VideoTrainer_Layer(images_paths=images_paths, num_points=args.num_points_3d,
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
    logwriter.write("{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        image_name, trainer.H, trainer.W, psnr, ms_ssim, training_time, eval_time, eval_fps))

    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    # avg_h = image_h//image_length
    # avg_w = image_w//image_length

    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        image_h, image_w, avg_psnr, avg_ms_ssim, avg_training_time, avg_eval_time, avg_eval_fps))    

if __name__ == "__main__":
    main(sys.argv[1:])