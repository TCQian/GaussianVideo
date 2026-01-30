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
import torchvision.transforms as transforms
from train_3D2D import EarlyStopping

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class VideoTrainer:
    """Trains 2d+t gaussians to fit a video."""
    def __init__(
        self,
        images_paths: list[Path],
        num_points: int = 2000,
        model_name:str = "GaussianVideo",
        iterations:int = 30000,
        model_path = None,
        args = None,
        video_name: str = "Jockey",
        num_frames: int = 50,
        start_frame: int = 0,
    ):
        self.early_stopping_patience = 1000
        self.early_stopping_min_delta = 1e-9

        self.video_name = video_name
        self.device = torch.device("cuda:0")
        self.gt_image = images_paths_to_tensor(images_paths).to(self.device)

        self.num_points = num_points
        BLOCK_H, BLOCK_W, BLOCK_T = 16, 16, 1
        self.H, self.W, self.T = self.gt_image.shape[2], self.gt_image.shape[3], self.gt_image.shape[4]
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.log_dir = Path(f"./checkpoints/{args.data_name}/{model_name}_i{args.iterations_3d}_g{num_points}_f{num_frames}_s{start_frame}/{video_name}")
            
        if model_name == "GaussianVideo":
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
        self.early_stopping = EarlyStopping(patience=self.early_stopping_patience, min_delta=self.early_stopping_min_delta)

        start_time = time.time()
        for iter in range(1, self.iterations+1):
            if iter == 1 or iter % 1000 == 0:
                self.gaussian_model.debug_mode = True
            else:
                self.gaussian_model.debug_mode = False

            if (iter % 1000 == 1 and iter > 1):
                self.gaussian_model.prune(opac_threshold=0.05)

            loss, psnr = self.gaussian_model.train_iter(self.gt_image)
            
            if self.early_stopping(loss.item()):
                print(f"Early stopping at iteration {iter}")
                break

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
