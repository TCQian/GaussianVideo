import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import math
import numpy as np
from gaussianvideo import GaussianVideo
from gaussianimage_cholesky import GaussianImage_Cholesky
from utils import *
from tqdm import tqdm
import time
import torchvision.transforms as transforms
import json
import yaml
import random
import sys

def get_kwargs(kwargs, layer):
    """Extract and validate parameters for specific layers from the JSON config"""
    
    # Ensure all required parameters are available with defaults
    required_params = {
        "device": kwargs.get("device", "cuda:0"),
        "H": kwargs.get("H", 1080),
        "W": kwargs.get("W", 1920),
        "BLOCK_H": kwargs.get("BLOCK_H", 16),
        "BLOCK_W": kwargs.get("BLOCK_W", 16),
        "dataset": kwargs.get("dataset", "./dataset/Jockey/"),
        "data_name": kwargs.get("data_name", "Jockey"),
        "num_frames": kwargs.get("num_frames", 50),
        "start_frame": kwargs.get("start_frame", 0),
        "seed": kwargs.get("seed", 1),
        "save_imgs": kwargs.get("save_imgs", False),
    }
    
    if layer == 0:  # 3D GaussianVideo (Layer 0)
        layer_kwargs = {
            **required_params,
            "model_name_3d": kwargs.get("model_name_3d", "GaussianVideo"),
            "iterations_3d": kwargs.get("iterations_3d", 50000),
            "num_points_3d": kwargs.get("num_points_3d", 50000),
            "lr_3d": kwargs.get("lr_3d", 1e-2),
            "T": kwargs.get("num_frames", 50),
            "BLOCK_T": kwargs.get("BLOCK_T", 16),
            "quantize": kwargs.get("quantize", False),
            "opt_type": kwargs.get("opt_type", "adan"),
        }
        
    elif layer == 1:  # 2D GaussianImage (Layer 1)
        layer_kwargs = {
            **required_params,
            "model_name_2d": kwargs.get("model_name_2d", "GaussianImage_Cholesky"),
            "iterations_2d": kwargs.get("iterations_2d", 50000),
            "num_points_2d": kwargs.get("num_points_2d", 50000),
            "lr_2d": kwargs.get("lr_2d", 1e-3),
            "quantize": kwargs.get("quantize", False),
            "opt_type": kwargs.get("opt_type", "adan"),
        }
        
    else:
        raise ValueError(f"Invalid layer: {layer}. Must be 0 (3D GaussianVideo) or 1 (2D GaussianImage)")
    
    # Filter out None values but keep all other values (including 0, False, etc.)
    layer_kwargs = {k: v for k, v in layer_kwargs.items() if v is not None}
    
    return layer_kwargs

def images_paths_to_tensor(images_paths: list[Path]):
    image_tensors = []
    
    for image_path in images_paths:
        img_tensor = image_path_to_tensor(image_path)
        image_tensors.append(img_tensor)
    
    stacked_tensor = torch.stack(image_tensors, dim=0)  # Shape: [T, 1, C, H, W]
    
    final_tensor = stacked_tensor.permute(1, 2, 3, 4, 0)  # Shape: [1, C, H, W, T]
    
    return final_tensor

def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0) #[1, C, H, W]
    return img_tensor

class Gaussian3Dplus2D(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.device = torch.device("cuda:0")
        self.iterations_3d = kwargs["iterations_3d"]
        self.iterations_2d = kwargs["iterations_2d"]
        self.iterations = self.iterations_3d + (self.iterations_2d * self.num_frames)

        self.start = kwargs["start"]
        self.num_frames = kwargs["num_frames"]

        self.log_dir = Path(f"./checkpoints/{kwargs['data_name']}/{kwargs['model_name_3d']}_i{kwargs['iterations_3d']}_g{kwargs['num_points_3d']}_{kwargs['model_name_2d']}_i{kwargs['iterations_2d']}_g{kwargs['num_points_2d']}_f{kwargs['num_frames']}_s{kwargs['start_frame']}/{kwargs['data_name']}")
        self.logwriter = LogWriter(self.log_dir)

        kwargs_0 = get_kwargs(kwargs, layer=0)
        self.layer_0_model = GaussianVideo(**kwargs_0)

        self.layer_1_models = []
        kwargs_1 = get_kwargs(kwargs, layer=1)
        for _ in range(self.num_frames):
            self.layer_1_models.append(GaussianImage_Cholesky(**kwargs_1))
        
        self.trained_3D = False
        self.trained_2D = False
        if kwargs.get("model_path_3d") is not None:
            checkpoint = torch.load(kwargs.get("model_path_3d"), map_location=self.device)
            model_dict = self.layer_0_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.layer_0_model.load_state_dict(model_dict)
            self.trained_3D = True

        if kwargs.get("model_path_2d") is not None:
            for t in range(self.num_frames):
                checkpoint = torch.load(kwargs.get("model_path_2d").replace(f"layer_1_model_*.pth.tar", f"layer_1_model_{t}.pth.tar"), map_location=self.device)
                model_dict = self.layer_1_models[t].state_dict()
                pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.layer_1_models[t].load_state_dict(model_dict)
            self.trained_2D = True
        
        images_paths = []
        for i in range(self.start, self.start+self.num_frames):
            image_path = Path(kwargs["dataset"]) / f'frame_{i+1:04}.png'
            images_paths.append(image_path)
        self.gt_image = images_paths_to_tensor(images_paths).to(self.device)

    def forward(self):
        assert self.trained_3D and self.trained_2D, "3D is trained: {}, 2D is trained: {}".format(self.trained_3D, self.trained_2D)
        self.layer_0_model.eval()
        for t in range(self.num_frames):
            self.layer_1_models[t].eval()
        
        log_str = "Test "
        psnr_per_layer = [0 for _ in range(2)]
        ms_ssim_per_layer = [0 for _ in range(2)]
        for layer in range(2):
            with torch.no_grad():
                if layer == 0 and self.trained_3D:
                    out = self.layer_0_model()
                    render_tensor = out["render"].float() 
                elif layer == 1 and self.trained_2D:
                    render_tensor = []
                    for t in range(self.num_frames):
                        out = self.layer_1_models[t]()
                        render_tensor.append(out["render"].float())
                    stacked_tensor = torch.stack(render_tensor, dim=0)  # Shape: [T, 1, H, W, C]
                    render_tensor = stacked_tensor.permute(1, 2, 3, 4, 0)  # Shape: [1, H, W, C, T]

                mse_loss = F.mse_loss(render_tensor.float(), self.gt_image.float())
                psnr = 10 * math.log10(1.0 / mse_loss.item())
                psnr_per_layer[layer] += psnr
            
            try:
                ms_ssim_values = []
                for t in range(self.num_frames):
                    # Extract the t-th frame from both render and ground truth
                    frame = render_tensor[..., t]  # e.g. shape: [1, 3, H, W]
                    gt_frame = self.gt_image[..., t] # e.g. shape: [1, 3, H, W]
                    # Attempt to compute MS-SSIM for this frame
                    ms_ssim_values.append(
                        ms_ssim(frame, gt_frame, data_range=1, size_average=True).item()
                    )

                ms_ssim_per_layer[layer] += sum(ms_ssim_values) / len(ms_ssim_values)
            except AssertionError as e:
                # In case the image is too small for ms-ssim, log the error and continue.
                self.logwriter.write("MS-SSIM could not be computed: " + str(e))
            
            if self.save_imgs:
                transform = transforms.ToPILImage()
                for t in range(self.num_frames):
                    img = render_tensor[0, :, :, :, t]  
                    pil_image = transform(img) 
                    name = f"{self.video_name}_fitting_t{t}_layer{layer}.png" 
                    pil_image.save(str(self.log_dir / name))
        
            log_str += "layer {}: PSNR:{:.4f}, MS_SSIM:{:.6f}".format(layer, psnr_per_layer[layer], ms_ssim_per_layer[layer])
        
        self.logwriter.write(log_str)
            
        return psnr_per_layer, ms_ssim_per_layer

    def train(self):
        assert not self.trained_3D and not self.trained_2D, "3D is trained: {}, 2D is trained: {}".format(self.trained_3D, self.trained_2D)

        total_time = 0
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        
        if not self.trained_3D:
            start_time = time.time()
            self.layer_0_model.train()
            for iter in range(1, self.iterations_3d+1):
                loss, psnr = self.layer_0_model.train_iter(self.gt_image)
                psnr_list.append(psnr)
                iter_list.append(iter)
                with torch.no_grad():
                    if iter % 10 == 0:
                        progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                        progress_bar.update(10)
            torch.save(self.layer_0_model.state_dict(), self.log_dir / "layer_0_model.pth.tar")
            total_time += time.time() - start_time
            self.trained_3D = True
        
        if not self.trained_2D:
            layer_0_img = self.layer_0_model()["render"] # background for L1 training
            for t in range(self.num_frames):
                background = layer_0_img[..., t].squeeze(0).permute(1, 2, 0)
                if background.device != self.device:
                    background = background.to(self.device)
                start_time = time.time()
                self.layer_1_models[t].set_background(background)
                self.layer_1_models[t].train()
                
                start_iter = self.iterations_3d + 1 + (t * self.iterations_2d)
                for iter in range(start_iter, start_iter+self.iterations_2d+1):
                    loss, psnr = self.layer_1_models[t].train_iter(self.gt_image[..., t])
                    psnr_list.append(psnr)
                    iter_list.append(iter)
                    with torch.no_grad():
                        if iter % 10 == 0:
                            progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                            progress_bar.update(10)
                total_time += time.time() - start_time
                torch.save(self.layer_1_models[t].state_dict(), self.log_dir / f"layer_1_model_{t}.pth.tar")
            self.trained_2D = True
        
        progress_bar.close()
        psnr_value, ms_ssim_value = self.test()
        with torch.no_grad():
            self.layer_0_model.eval()
            for t in range(self.num_frames):
                self.layer_1_models[t].eval()
            
            test_start_time = time.time()
            for i in range(100):
                _ = self.layer_0_model()
                for t in range(self.num_frames):
                    _ = self.layer_1_models[t]()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("3D + 2D Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(total_time, test_end_time, 1/test_end_time))
        np.save(self.log_dir / "training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": total_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time})
        return psnr_value, ms_ssim_value, total_time, test_end_time, 1/test_end_time

def load_config(config_path="config/gaussian3D2D.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from: {config_path}")
        
        # Validate required parameters
        required_params = [
            "dataset", "data_name", "num_frames", "start_frame",
            "iterations_3d", "num_points_3d", "lr_3d",
            "iterations_2d", "num_points_2d", "lr_2d"
        ]
        
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            print(f"Warning: Missing parameters in config: {missing_params}")
            print("Using default values for missing parameters.")
        
        return config
    except FileNotFoundError:
        print(f"Warning: Configuration file '{config_path}' not found. Using default values.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{config_path}': {e}")
        sys.exit(1)

def parse_args(argv):
    """Parse command line arguments or load from JSON config"""
    # Check if config file is provided as command line argument
    config_path = "config/gaussian3D2D.json"
    if len(argv) > 0 and argv[0].endswith('.json'):
        config_path = argv[0]
        argv = argv[1:] 
    
    try:
        config = load_config(config_path)
        args.update(config)
    except:
        args = {}
    
    i = 0
    while i < len(argv):
        if argv[i].startswith('--'):
            key = argv[i][2:]  # Remove '--'
            if i + 1 < len(argv) and not argv[i + 1].startswith('--'):
                value = argv[i + 1]
                # Try to convert to appropriate type
                if key in ['seed', 'num_frames', 'start_frame', 'iterations_3d', 'num_points_3d', 
                          'iterations_2d', 'num_points_2d', 'H', 'W', 'BLOCK_H', 'BLOCK_W', 'T', 'BLOCK_T']:
                    args[key] = int(value)
                elif key in ['lr_3d', 'lr_2d']:
                    args[key] = float(value)
                elif key in ['save_imgs', 'quantize']:
                    args[key] = value.lower() in ['true', '1', 'yes']
                else:
                    args[key] = value
                i += 2
            else:
                # Boolean flag
                args[key] = True
                i += 1
        else:
            i += 1
    
    return args

def main(argv):
    # Parse arguments and get config as a dictionary
    config = parse_args(argv)

    # Print the configuration being used
    print("Configuration loaded:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    if config.get("seed") is not None:
        torch.manual_seed(config["seed"])
        random.seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(config["seed"])
    
    # Pass the config dictionary directly to the model
    print("Initializing Gaussian3Dplus2D model...")
    gaussian3D2D = Gaussian3Dplus2D(**config)
    print("Starting training...")
    gaussian3D2D.train()
    print("Running model...")
    gaussian3D2D()
    

if __name__ == "__main__":
    main(sys.argv[1:])

