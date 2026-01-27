import torch
import numpy as np
import math
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
from gaussianimage_cholesky import GaussianImage_Cholesky
from gaussianvideo3D2D import GaussianVideo3D2D

class EarlyStopping:
    def __init__(self, patience=100, min_delta=1e-10):
        self.patience = patience  # Number of tolerated iterations with no improvement
        self.min_delta = min_delta  # Minimum improvement threshold
        self.best_loss = None  # Stores the best loss value
        self.counter = 0  # Tracks the number of iterations without improvement

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            return False  # Do not stop training

        # If the improvement over the previous best loss is less than min_delta, consider it no improvement
        if self.best_loss - current_loss > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0  # Reset counter
        else:
            self.counter += 1

        # If the counter exceeds patience, stop training
        if self.counter >= self.patience:
            return True  # Stop training

        return False  # Continue training

class GaussianVideo3D2DTrainer:
    """Trains 3D and 2D gaussians layer by layer to fit a video."""
    def __init__(
        self,
        layer: int,
        images_paths: list[Path],
        test_images_paths: list[Path],
        model_name:str = "GV3D2D",
        args = None,
        video_name: str = "Jockey",
        num_frames: int = 50,
        start_frame: int = 0,
        log_dir: Path = None,
    ):

        self.layer = layer
        self.video_name = video_name
        self.model_name = model_name
        self.device = torch.device("cuda:0")
        self.gt_image = images_paths_to_tensor(images_paths).to(self.device)
        self.test_gt_image = images_paths_to_tensor(test_images_paths).to(self.device)

        BLOCK_H, BLOCK_W, BLOCK_T = 16, 16, 1
        self.H, self.W, self.T = self.gt_image.shape[2], self.gt_image.shape[3], self.gt_image.shape[4]
        self.iterations = args.iterations
        self.num_points = args.num_points
        self.save_imgs = args.save_imgs
        self.log_dir = log_dir

        self.early_stopping_patience = 1000
        self.early_stopping_min_delta = 1e-10
        self.densify_until_iter = self.iterations - 5000
        self.densify_factor = 0.1
        
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
                quantize=False,
                num_points=self.num_points,
                iterations=self.iterations,
                lr=args.lr
            )
            self.gaussian_model._create_data_from_checkpoint(args.model_path_layer0, args.model_path_layer1)
            self.gaussian_model.to(self.device)

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
                    quantize=False,
                    iterations=self.iterations,
                    lr=args.lr
                ).to(self.device)

                if args.model_path_layer1:
                    model_dict = gaussian_model.state_dict()
                    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    gaussian_model.load_state_dict(model_dict, strict=False)
                    gaussian_model._init_data()
                    print(f"Loaded checkpoint from: {checkpoint_file_path}")

                self.gaussian_model_list.append(gaussian_model)

        self.logwriter = LogWriter(self.log_dir)

    def train_GV3D2D(self, gaussian_model, gt_image):
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        best_psnr = 0
        gaussian_model.train()
        self.early_stopping = EarlyStopping(patience=self.early_stopping_patience, min_delta=self.early_stopping_min_delta)
        
        start_time = time.time()
        for iter in range(1, self.iterations+1):
            if iter == 1 or iter % 1000 == 0:
                gaussian_model.debug_mode = True
            else:
                gaussian_model.debug_mode = False

            if (iter % 1000 == 1 and iter > 1):
                gaussian_model.prune(opac_threshold=0.05)
                # if self.layer == 1 and iter < self.densify_until_iter:
                #     num_new_gaussians = max(100, int(gaussian_model._xyz_2D.shape[0] * self.densify_factor))
                #     gaussian_model.densify(num_new_gaussians=num_new_gaussians)

            loss, psnr = gaussian_model.train_iter(gt_image)
            
            if self.early_stopping(loss.item()):
                print(f"Early stopping at iteration {iter}")
                break

            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
        print(f"Number of gaussians at the end of training: {gaussian_model.get_xyz.shape[0]}")
        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value, gt_psnr_value, gt_ms_ssim_value = self.test()
        with torch.no_grad():
            gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = gaussian_model()
            test_end_time = (time.time() - test_start_time)/100
        FPS = self.T/test_end_time
        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, FPS))
        gaussian_model.save_checkpoint(self.log_dir)
        np.save(self.log_dir / f"training_layer_{gaussian_model.layer}.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
            "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": FPS})
        
        return psnr_value, ms_ssim_value, end_time, test_end_time, FPS, gt_psnr_value, gt_ms_ssim_value

    def train_GVGI(self, gaussian_model, gt_image, t=0):     
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        best_psnr = 0
        gaussian_model.train()
        self.early_stopping = EarlyStopping(patience=self.early_stopping_patience, min_delta=self.early_stopping_min_delta)

        start_time = time.time()
        for iter in range(1, self.iterations+1):
            if iter == 1 or iter % 1000 == 0:
                gaussian_model.debug_mode = True
            else:
                gaussian_model.debug_mode = False

            if (iter % 1000 == 1 and iter > 1):
                gaussian_model.prune(opac_threshold=0.05)
                # if self.layer == 1 and iter < self.densify_until_iter:
                #     num_new_gaussians = max(100, int(gaussian_model.get_xyz.shape[0] * self.densify_factor))
                #     gaussian_model.densify(num_new_gaussians=num_new_gaussians)

            loss, psnr = gaussian_model.train_iter(gt_image)

            if self.early_stopping(loss.item()):
                print(f"Early stopping at iteration {iter}")
                break

            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
        print(f"Number of gaussians at the end of training: {gaussian_model.get_xyz.shape[0]}")
        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value = self.test_GVGI(gaussian_model, self.test_gt_image[...,t], t)
        gt_psnr_value, gt_ms_ssim_value = self.test_GVGI(gaussian_model, self.gt_image[...,t], t)
        with torch.no_grad():
            gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = gaussian_model()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        
        # remove background from state_dict
        state_dict = gaussian_model.state_dict()
        state_dict.pop('background')
        torch.save(state_dict, self.log_dir / f'frame_{t+1:04}' / f"gaussian_model.pth.tar")
        np.save(self.log_dir / f'frame_{t+1:04}' / f"training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time})
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time, gt_psnr_value, gt_ms_ssim_value

    def train(self):  
        if self.layer == 0 or self.model_name == "GV3D2D":
            psnr_value, ms_ssim_value, end_time, test_end_time, eval_fps, gt_psnr_value, gt_ms_ssim_value = self.train_GV3D2D(self.gaussian_model, self.gt_image)
        elif self.layer == 1:
            psnr_value_list = []
            ms_ssim_value_list = []
            end_time_list = []
            test_end_time_list = []
            eval_fps_list = []
            gt_psnr_value_list = []
            gt_ms_ssim_value_list = []
            for t in range(self.T):
                psnr_value, ms_ssim_value, end_time, test_end_time, eval_fps, gt_psnr_value, gt_ms_ssim_value = self.train_GVGI(self.gaussian_model_list[t], self.gt_image[...,t], t)
                psnr_value_list.append(psnr_value)
                ms_ssim_value_list.append(ms_ssim_value)
                end_time_list.append(end_time)
                test_end_time_list.append(test_end_time)
                eval_fps_list.append(eval_fps)
                gt_psnr_value_list.append(gt_psnr_value)
                gt_ms_ssim_value_list.append(gt_ms_ssim_value)
        
            psnr_value = sum(psnr_value_list) / len(psnr_value_list)
            ms_ssim_value = sum(ms_ssim_value_list) / len(ms_ssim_value_list)
            end_time = sum(end_time_list) / len(end_time_list)
            test_end_time = sum(test_end_time_list) / len(test_end_time_list)
            eval_fps = sum(eval_fps_list) / len(eval_fps_list)
            gt_psnr_value = sum(gt_psnr_value_list) / len(gt_psnr_value_list)
            gt_ms_ssim_value = sum(gt_ms_ssim_value_list) / len(gt_ms_ssim_value_list)
        return psnr_value, ms_ssim_value, end_time, test_end_time, eval_fps, gt_psnr_value, gt_ms_ssim_value

    def test_GV3D2D(self, gaussian_model, gt_image):
        gaussian_model.eval()
        with torch.no_grad():
            out = gaussian_model()
        mse_loss = F.mse_loss(out["render"].float(), gt_image.float())
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
                gt_frame = gt_image[..., t] # e.g. shape: [1, 3, H, W]
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
                name = f"{self.video_name}_fitting_t{t}_layer{self.layer}.png"
                pil_image.save(str(self.log_dir / name))
        return psnr, ms_ssim_value

    def test_GVGI(self, gaussian_model, gt_image, t=0):
        gaussian_model.eval()
        with torch.no_grad():
            out = gaussian_model()
        mse_loss = F.mse_loss(out["render"].float(), gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out["render"].float(), gt_image.float(), data_range=1, size_average=True).item()
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
        if self.save_imgs:
            transform = transforms.ToPILImage()
            img = transform(out["render"].float().squeeze(0))
            name = f"{self.video_name}_fitting_t{t}_layer{self.layer}.png"
            Path(self.log_dir / f'frame_{t+1:04}').mkdir(parents=True, exist_ok=True)
            img.save(str(self.log_dir / f'frame_{t+1:04}' / name))
        return psnr, ms_ssim_value

    def test(self):
        if self.layer == 0 or self.model_name == "GV3D2D":
            psnr_value, ms_ssim_value = self.test_GV3D2D(self.gaussian_model, self.test_gt_image)
            gt_psnr_value, gt_ms_ssim_value = self.test_GV3D2D(self.gaussian_model, self.gt_image)
        elif self.layer == 1:
            psnr_value_list = []
            ms_ssim_value_list = []
            gt_psnr_value_list = []
            gt_ms_ssim_value_list = []
            for t in range(self.T):
                psnr_value, ms_ssim_value = self.test_GVGI(self.gaussian_model_list[t], self.test_gt_image[...,t], t)
                gt_psnr_value, gt_ms_ssim_value = self.test_GVGI(self.gaussian_model_list[t], self.gt_image[...,t], t)
                psnr_value_list.append(psnr_value)
                ms_ssim_value_list.append(ms_ssim_value)
                gt_psnr_value_list.append(gt_psnr_value)
                gt_ms_ssim_value_list.append(gt_ms_ssim_value)
            psnr_value = sum(psnr_value_list) / len(psnr_value_list)
            ms_ssim_value = sum(ms_ssim_value_list) / len(ms_ssim_value_list)
        return psnr_value, ms_ssim_value, gt_psnr_value, gt_ms_ssim_value

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train GaussianVideo model layer by layer.")
    
    # Parameters for training
    parser.add_argument("--seed", type=int, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument("--downsampling_factor", type=float, default=0.25, help="Downsampling factor (default: %(default)s)")
    
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

    log_dir = Path(f"./checkpoints/{args.data_name}/ProgressiveGaussianVideo_i{args.iterations}_g{args.num_points}_f{args.num_frames}_s{args.start_frame}_L0x{args.downsampling_factor}/layer{args.layer}/")
    if args.layer == 1:
        log_dir = log_dir / (f"{args.model_name}_i{args.iterations}_g{args.num_points}/")

    logwriter = LogWriter(log_dir)
    image_length, start = args.num_frames, args.start_frame

    images_paths = []
    test_images_paths = []
    if args.layer == 0:
        downsampled_dataset_path = os.path.join(os.path.dirname(args.dataset), f'{args.data_name}_downsampled_{args.downsampling_factor}/')
        os.makedirs(downsampled_dataset_path, exist_ok=True)
        for i in range(start, start+image_length):
            image_path = Path(args.dataset) / f'frame_{i+1:04}.png'
            downsampled_image_path = downsampled_dataset_path / f'frame_{i+1:04}.png'
            
            image = Image.open(image_path)
            transform = transforms.ToTensor()
            image_tensor = transform(image).unsqueeze(0)
            _, _, H, W = image_tensor.shape
            
            downsampled = F.interpolate(image_tensor, scale_factor=args.downsampling_factor, mode='area')
            resized_back = F.interpolate(downsampled, size=(H, W), mode='bilinear', align_corners=False)
            resized_back = resized_back.squeeze(0)
            transform_to_pil = transforms.ToPILImage()
            blurred_image = transform_to_pil(resized_back)
            
            os.makedirs(os.path.dirname(downsampled_image_path), exist_ok=True)
            blurred_image.save(downsampled_image_path)

            images_paths.append(downsampled_image_path)
            test_images_paths.append(image_path)
    elif args.layer == 1:
        for i in range(start, start+image_length):
            image_path = Path(args.dataset) / f'frame_{i+1:04}.png'
            images_paths.append(image_path)
            test_images_paths.append(image_path)

    trainer = GaussianVideo3D2DTrainer(layer=args.layer, images_paths=images_paths, test_images_paths=test_images_paths, model_name=args.model_name, args=args, num_frames=args.num_frames, start_frame=args.start_frame, video_name=args.data_name, log_dir=log_dir)
    psnr, ms_ssim, training_time, eval_time, eval_fps, gt_psnr, gt_ms_ssim = trainer.train()

    logwriter.write("Average Rendered VS Train_GT: {}x{} for layer {}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        trainer.H, trainer.H, args.layer, gt_psnr, gt_ms_ssim, training_time, eval_time, eval_fps))    
    logwriter.write("Average: {}x{} for layer {}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        trainer.H, trainer.H, args.layer, psnr, ms_ssim, training_time, eval_time, eval_fps))    

if __name__ == "__main__":
    main(sys.argv[1:])