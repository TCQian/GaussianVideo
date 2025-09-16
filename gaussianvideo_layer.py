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
from utils import *
class GaussianVideo_Layer(nn.Module):
    def __init__(self, layer= 0, loss_type="L2", **kwargs):
        super().__init__()

        self.debug_mode = False 

        self.layer = int(layer)
        self.loss_type = loss_type
        self.opt_type = kwargs["opt_type"]
        self.lr = kwargs["lr"]
        self.init_num_points_3D = kwargs["num_points_layer0"]
        self.init_num_points_2D = kwargs["num_points_layer1"]

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
        self.register_buffer('cholesky_bound_2D', torch.tensor([0.5, 0, 0, 0.5, 0, 0]).view(1, 6))

        self.opacity_activation = torch.sigmoid

        self.checkpoint_path = kwargs.get("checkpoint_path", None)

        self._init_layer0() if self.layer == 0 else self._init_layer1()

        if self.layer == 1:
            self._setup_progressive_optimizer()
        else:
            if self.opt_type == "adam":
                self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            else:
                self.optimizer = Adan(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    def _init_layer0(self):
        self._xyz_3D = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points_3D, 3) - 0.5)))
        self._cholesky_3D = nn.Parameter(torch.rand(self.init_num_points_3D, 6))
        self._opacity_3D = nn.Parameter(torch.logit(0.1 * torch.ones(self.init_num_points_3D, 1)))
        self._features_dc_3D = nn.Parameter(torch.rand(self.init_num_points_3D, 3))
        
        # Increase L33 (the last element in each row) to boost temporal variance.
        with torch.no_grad():
            self._cholesky_3D.data[:, 5] += self.T  # adjust the constant as needed

        self.layer = 0
        print("GaussianVideo_Layer: Layer 0 initialized, number of parameters: ", sum(p.numel() for p in self.parameters()))

    def _init_layer1(self):
        assert self.checkpoint_path is not None, "GaussianVideo_Layer: Layer 1 requires a layer 0 checkpoint"
        self._load_layer0_checkpoint()

        # check if _opacity_3D is a parameter
        if not isinstance(self._opacity_3D, nn.Parameter):
            print("GaussianVideo_Layer: _opacity_3D is not a parameter, creating parameter")
            self._opacity_3D = nn.Parameter(torch.logit(0.1 * torch.ones(self.init_num_points_3D, 1)))

        self._xyz_2D = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points_2D * self.T, 3) - 0.5)))
        if self.T > 1:
            for t in range(self.T):
                val = torch.atanh(torch.tensor(2 * (t / (self.T - 1)) - 1.0)).item()
                self._xyz_2D.data[t * self.init_num_points_2D : (t + 1) * self.init_num_points_2D, 2] = val
        else:
            self._xyz_2D.data[:, 2] = torch.atanh(torch.tensor(- 1.0)).item()

        # 2D -> 3D, L11, L12, L22 are random, L13, L23 are 0 and L33 is 1
        self._cholesky_2D = nn.Parameter(torch.rand(self.init_num_points_2D  * self.T, 6))
        with torch.no_grad():
            self._cholesky_2D.data[:, 2] = 0
            self._cholesky_2D.data[:, 4] = 0
            self._cholesky_2D.data[:, 5] = 1

        self._opacity_2D = nn.Parameter(torch.logit(0.1 * torch.ones(self.init_num_points_2D * self.T, 1)))
        # self.register_buffer('_opacity_2D', torch.ones((self.init_num_points_2D * self.T, 1)))
        self._features_dc_2D = nn.Parameter(torch.rand(self.init_num_points_2D * self.T, 3))
        self.layer = 1
        print("GaussianVideo_Layer: Layer 1 initialized, number of parameters: ", sum(p.numel() for p in self.parameters()))

    def _load_layer0_checkpoint(self):
        """Load layer 0 checkpoint for progressive training"""
        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided for progressive training")
        
        print(f"Loading layer 0 checkpoint from: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        self._init_layer0()
        layer0_params = {
            '_xyz_3D': checkpoint.get('_xyz_3D'),
            '_cholesky_3D': checkpoint.get('_cholesky_3D'),
            '_opacity_3D': checkpoint.get('_opacity_3D'),
            '_features_dc_3D': checkpoint.get('_features_dc_3D')
        }
        
        for param_name, param_value in layer0_params.items():
            if param_value is not None and hasattr(self, param_name):
                if param_name == '_opacity_3D':
                    self._opacity_3D = nn.Parameter(param_value.clone())
                else:
                    getattr(self, param_name).data.copy_(param_value)
        
        print("Layer 0 checkpoint loaded successfully")

    def _freeze_layer0_parameters(self):
        """Freeze layer 0 parameters and apply 2D gaussian constraints"""
        if self.layer == 1:
            # Freeze layer 0 parameters by setting requires_grad=False
            layer0_params = ['_xyz_3D', '_cholesky_3D', '_features_dc_3D']
            
            for param_name in layer0_params:
                if hasattr(self, param_name):
                    param = getattr(self, param_name)
                    if isinstance(param, nn.Parameter):
                        param.requires_grad = False
                        print(f"Frozen layer 0 parameter: {param_name}")

    def _apply_2d_gaussian_constraints(self):
        """Apply gradient constraints to 2D gaussian parameters during training"""
        if self.layer == 1:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    if name == "_xyz_2D":
                        # Freeze temporal dimension (z-coordinate)
                        param.grad[:, 2] = 0
                    elif name == "_cholesky_2D":
                        # Freeze L13, L23, L33 elements (indices 2, 4, 5)
                        param.grad[:, [2, 4, 5]] = 0

    def _setup_progressive_optimizer(self):
        """Setup optimizer for progressive training - only train 2D gaussians and extra 3D features"""
        self._freeze_layer0_parameters()
        
        trainable_params = []
        trainable_params.extend([
            self._xyz_2D,
            self._cholesky_2D,
            self._features_dc_2D,
            self._opacity_2D
        ])
        
        if hasattr(self, '_opacity_3D') and isinstance(self._opacity_3D, nn.Parameter) and self._opacity_3D.requires_grad:
            trainable_params.append(self._opacity_3D)
        
        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr)
        else:
            self.optimizer = Adan(trainable_params, lr=self.lr)
        
        print(f"Progressive optimizer setup: training {len(trainable_params)} parameter groups")

    def save_checkpoint(self, path):
        if self.layer == 0:
            # Save only layer 0 parameters (3D gaussians)
            layer0_state = {
                '_xyz_3D': self._xyz_3D.data,
                '_cholesky_3D': self._cholesky_3D.data,
                '_opacity_3D': self._opacity_3D.data if isinstance(self._opacity_3D, nn.Parameter) else self._opacity_3D,
                '_features_dc_3D': self._features_dc_3D.data,
                'layer': self.layer
            }
            torch.save(layer0_state, path / "layer_0_model.pth.tar")
            print(f"Layer 0 checkpoint saved to: {path / 'layer_0_model.pth.tar'}")
        elif self.layer == 1:
            layer1_state = {
                '_xyz_2D': self._xyz_2D.data[:, :2],
                '_cholesky_2D': self._cholesky_2D.data[:, [0, 1, 3]],
                '_features_dc_2D': self._features_dc_2D.data,
                '_opacity_2D': self._opacity_2D.data if isinstance(self._opacity_2D, nn.Parameter) else self._opacity_2D,
                '_opacity_3D': self._opacity_3D.data if isinstance(self._opacity_3D, nn.Parameter) else self._opacity_3D,
                'layer': self.layer
            }
            torch.save(layer1_state, path / "layer_1_model.pth.tar")
            print(f"Layer 1 checkpoint saved to: {path / 'layer_1_model.pth.tar'}")

    @property
    def get_xyz(self):
        if self.layer == 0:
            return torch.tanh(self._xyz_3D)
        elif self.layer == 1:
            if self.debug_mode:
                print("Gaussian's xyz coordintes", self._xyz_2D[:, 2])
            return torch.tanh(torch.cat((self._xyz_3D, self._xyz_2D), dim=0))
    
    @property
    def get_features(self):
        if self.layer == 0:
            return self._features_dc_3D
        elif self.layer == 1:
            return torch.cat((self._features_dc_3D, self._features_dc_2D), dim=0)
    
    @property
    def get_opacity(self):
        if self.layer == 0:
            return self.opacity_activation(self._opacity_3D)
        elif self.layer == 1:
            return self.opacity_activation(torch.cat((self._opacity_3D, self._opacity_2D), dim=0))
    
    @property
    def get_cholesky_elements(self):
        if self.layer == 0:
            return self._cholesky_3D + self.cholesky_bound_3D
        elif self.layer == 1:
            if self.debug_mode:
                print("Gaussian's cholesky elements", self._cholesky_2D[:, [2, 4, 5]])
            merged_cholesky = torch.cat((self._cholesky_3D, self._cholesky_2D), dim=0)
            merged_bounds = torch.cat((self.cholesky_bound_3D, self.cholesky_bound_2D), dim=0)
            return merged_cholesky + merged_bounds
    
    def forward(self):
        self.xys, depths, radii, conics, num_tiles_hit = project_gaussians_video(
            self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.T, self.tile_bounds
        )
        out_img = rasterize_gaussians_sum_video(
            self.xys, depths, radii, conics, num_tiles_hit,
            self.get_features, self.get_opacity, self.H, self.W, self.T,
            self.BLOCK_H, self.BLOCK_W, self.BLOCK_T,
            background=self.background, return_alpha=False
        )
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
            self._apply_2d_gaussian_constraints()

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        
        return loss, psnr


class ProgressiveVideoTrainer:
    """Trains 3D and 2D gaussians layer by layer to fit a video."""
    def __init__(
        self,
        layer: int,
        images_paths: list[Path],
        model_name:str = "GaussianVideo_Layer",
        args = None,
        video_name: str = "Jockey",
        num_frames: int = 50,
        start_frame: int = 0,
    ):

        self.video_name = video_name
        self.device = torch.device("cuda:0")
        self.gt_image = images_paths_to_tensor(images_paths).to(self.device)

        self.num_points_layer0 = args.num_points_layer0
        self.num_points_layer1 = args.num_points_layer1
        BLOCK_H, BLOCK_W, BLOCK_T = 16, 16, 1
        self.H, self.W, self.T = self.gt_image.shape[2], self.gt_image.shape[3], self.gt_image.shape[4]
        self.iterations_layer0 = args.iterations_layer0
        self.iterations_layer1 = args.iterations_layer1
        self.save_imgs = args.save_imgs
        self.log_dir = Path(f"./checkpoints/{args.data_name}/{model_name}_i{args.iterations_layer0}+{args.iterations_layer1}_g{args.num_points_layer0}+{args.num_points_layer1}_f{num_frames}_s{start_frame}/{video_name}")
            
        if model_name == "GaussianVideo_Layer":
            checkpoint_path = None
            if layer == 1:
                assert args.model_path_layer0 is not None, "GaussianVideo_Layer: Layer 1 requires a layer 0 checkpoint"
                checkpoint_path = args.model_path_layer0
                
            self.iterations = self.iterations_layer0 if layer == 0 else self.iterations_layer1

            self.gaussian_model = GaussianVideo_Layer(
                layer=layer,
                loss_type="L2", 
                opt_type="adan", 
                H=self.H, 
                W=self.W, 
                T=self.T, 
                BLOCK_H=BLOCK_H, 
                BLOCK_W=BLOCK_W, 
                BLOCK_T=BLOCK_T, 
                device=self.device, 
                quantize=False,
                checkpoint_path=checkpoint_path,
                num_points_layer0=self.num_points_layer0,
                num_points_layer1=self.num_points_layer1,
                lr=args.lr_layer0 if layer == 0 else args.lr_layer1
            ).to(self.device)

        self.logwriter = LogWriter(self.log_dir)

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
        
        # Save checkpoint using the model's save_checkpointsave_checkpoint method
        self.gaussian_model.save_checkpoint(self.log_dir)
        
        np.save(self.log_dir / "training_layer_{}.npy".format(self.gaussian_model.layer), {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
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
                img = render_tensor[0, :, :, :, t]  # Shape: [C, H, W]
                pil_image = transform(img)  # Convert to PIL Image
                name = f"{self.video_name}_fitting_t{t}.png"  # e.g., "_fitting_t0.png"
                pil_image.save(str(self.log_dir / name))
        return psnr, ms_ssim_value


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    
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
        "--model_name", type=str, default="GaussianVideo_Layer", help="model selection: GaussianVideo_Layer"
    )

    parser.add_argument(
        "--iterations_layer0", type=int, default=50000, help="number of training epochs for layer 0 (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points_layer0",
        type=int,
        default=50000,
        help="3D GS points in layer 0 (default: %(default)s)",
    )
    parser.add_argument(
        "--lr_layer0",
        type=float,
        default=1e-2,
        help="Learning rate of layer 0 (default: %(default)s)",
    )
    parser.add_argument("--model_path_layer0", type=str, default=None, help="Path to a layer 0 checkpoint")

    # Model parameters for layer 1
    parser.add_argument(
        "--iterations_layer1", type=int, default=50000, help="number of training epochs for layer 1 (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points_layer1",
        type=int,
        default=50000,
        help="2D GS points per frame in layer 1 (default: %(default)s)",
    )
    parser.add_argument(
        "--lr_layer1",
        type=float,
        default=1e-3,
        help="Learning rate for layer 1 (default: %(default)s)",
    )
    parser.add_argument("--model_path_layer1", type=str, default=None, help="Path to a layer 1 checkpoint")
    
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

    logwriter = LogWriter(Path(f"./checkpoints/{args.data_name}/{args.model_name}_i{args.iterations_layer0}+{args.iterations_layer1}_g{args.num_points_layer0}+{args.num_points_layer1}_f{args.num_frames}_s{args.start_frame}"))
    image_length, start = args.num_frames, args.start_frame

    images_paths = []
    for i in range(start, start+image_length):
        image_path = Path(args.dataset) / f'frame_{i+1:04}.png'
        images_paths.append(image_path)

    trainer = ProgressiveVideoTrainer(layer=args.layer, images_paths=images_paths, model_name=args.model_name, args=args, num_frames=args.num_frames, start_frame=args.start_frame, video_name=args.data_name)
    psnr, ms_ssim, training_time, eval_time, eval_fps = trainer.train()

    logwriter.write("Average: {}x{} for layer {}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        trainer.H, trainer.H, args.layer, psnr, ms_ssim, training_time, eval_time, eval_fps))    

if __name__ == "__main__":
    main(sys.argv[1:])