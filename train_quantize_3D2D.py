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

from utils import images_paths_to_tensor
from gaussianvideo3D2D import GaussianVideo3D2D
from gaussianimage_cholesky import GaussianImage_Cholesky
from train_3D2D import EarlyStopping

class GaussianVideo3D2DTrainerQuantize:
    """Trains 3D and 2D gaussians layer by layer with quantization to fit a video."""
    def __init__(
        self,
        layer: int,
        images_paths: list[Path],
        model_name:str = "GV3D2D",
        args = None,
        video_name: str = "Jockey",
        num_frames: int = 50,
        start_frame: int = 0,
        log_dir: Path = None,
    ):

        self.early_stopping = EarlyStopping(patience=1000, min_delta=1e-10)

        self.device = torch.device("cuda:0")
        self.gt_image = images_paths_to_tensor(images_paths).to(self.device)  # [1, C, H, W, T]
        self.video_name = video_name
        self.model_name = model_name
        self.layer = layer

        # Determine spatial and temporal dimensions
        self.H, self.W, self.T = self.gt_image.shape[2], self.gt_image.shape[3], self.gt_image.shape[4]
        BLOCK_H, BLOCK_W, BLOCK_T = 16, 16, 1  # adjust BLOCK_T if needed
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

        self.logwriter = LogWriter(self.log_dir)

    def train_GV3D2D(self, gaussian_model, gt_image):
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        best_psnr = 0
        gaussian_model.train()
        start_time = time.time()
        for iter in range(1, self.iterations+1):
            if iter == 1 or iter % 1000 == 0:
                gaussian_model.debug_mode = True
            else:
                gaussian_model.debug_mode = False

            if (iter % 1000 == 1 and iter > 1):
                gaussian_model.prune(opac_threshold=0.05)

            loss, psnr = gaussian_model.train_iter_quantize(gt_image)
            
            if self.early_stopping(loss.item()):
                print(f"Early stopping at iteration {iter}")
                break

            psnr_list.append(psnr)
            iter_list.append(iter)
            if psnr > best_psnr:
                best_psnr = psnr
                best_model_dict = copy.deepcopy(gaussian_model.get_state_dict())
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
        gaussian_model.save_checkpoint(self.log_dir)
        print(f"Number of gaussians at the end of training: {gaussian_model.get_xyz.shape[0]}")
        
        gaussian_model.load_state_dict(best_model_dict)
        best_psnr_value, best_ms_ssim_value, best_bpp = self.test(best=True)
        gaussian_model.save_checkpoint(self.log_dir, best=True)

        with torch.no_grad():
            gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = gaussian_model.forward_quantize()
            test_end_time = (time.time() - test_start_time) / 100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}"
                             .format(end_time, test_end_time, 1/test_end_time))
        np.save(self.log_dir / f"training_layer_{gaussian_model.layer}.npy", {
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

    def train_GVGI(self, gaussian_model, gt_image, t=0):     
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        best_psnr = 0
        gaussian_model.train()
        start_time = time.time()
        for iter in range(1, self.iterations+1):
            if iter == 1 or iter % 1000 == 0:
                gaussian_model.debug_mode = True
            else:
                gaussian_model.debug_mode = False

            if (iter % 1000 == 1 and iter > 1):
                gaussian_model.prune(opac_threshold=0.05)

            loss, psnr = gaussian_model.train_iter_quantize(gt_image)

            if self.early_stopping(loss.item()):
                print(f"Early stopping at iteration {iter}")
                break

            psnr_list.append(psnr)
            iter_list.append(iter)
            if best_psnr < psnr:
                best_psnr = psnr
                best_model_dict = copy.deepcopy(gaussian_model.state_dict())
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f}", "Best PSNR":f"{best_psnr:.{4}f}"})
                    progress_bar.update(10)

        print(f"Number of gaussians at the end of training: {gaussian_model.get_xyz.shape[0]}")
        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value, bpp = self.test()
        torch.save(self.gaussian_model.state_dict(), self.log_dir / f'frame_{t+1:04}' / f"gaussian_model.pth.tar")
        self.gaussian_model.load_state_dict(best_model_dict)
        best_psnr_value, best_ms_ssim_value, best_bpp = self.test(True)
        torch.save(best_model_dict, self.log_dir / self.log_dir / f'frame_{t+1:04}' / f"gaussian_model.best.pth.tar")
        
        with torch.no_grad():
            gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = gaussian_model.forward_quantize()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        np.save(self.log_dir / f"training_layer_{gaussian_model.layer}.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time, "bpp":bpp, 
        "best_psnr":best_psnr_value, "best_ms-ssim":best_ms_ssim_value, "best_bpp": best_bpp})
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time, bpp, best_psnr_value, best_ms_ssim_value, best_bpp

    def train(self):  
        if self.model_name == "GV3D2D":
            psnr_value, ms_ssim_value, end_time, test_end_time, eval_fps, bpp, best_psnr_value, best_ms_ssim_value, best_bpp = self.train_GV3D2D(self.gaussian_model, self.gt_image)
        elif self.model_name == "GVGI":
            psnr_value_list = []
            ms_ssim_value_list = []
            end_time_list = []
            test_end_time_list = []
            eval_fps_list = []
            bpp_list = []
            best_psnr_value_list = []
            best_ms_ssim_value_list = []
            best_bpp_list = []
            for t in range(self.T):
                psnr_value, ms_ssim_value, end_time, test_end_time, eval_fps, bpp, best_psnr_value, best_ms_ssim_value, best_bpp = self.train_GVGI(self.gaussian_model_list[t], self.gt_image[...,t], t)
                psnr_value_list.append(psnr_value)
                ms_ssim_value_list.append(ms_ssim_value)
                end_time_list.append(end_time)
                test_end_time_list.append(test_end_time)
                eval_fps_list.append(eval_fps)
                bpp_list.append(bpp)
                best_psnr_value_list.append(best_psnr_value)
                best_ms_ssim_value_list.append(best_ms_ssim_value)
                best_bpp_list.append(best_bpp)
            psnr_value = sum(psnr_value_list) / len(psnr_value_list)
            ms_ssim_value = sum(ms_ssim_value_list) / len(ms_ssim_value_list)
            end_time = sum(end_time_list) / len(end_time_list)
            test_end_time = sum(test_end_time_list) / len(test_end_time_list)
            eval_fps = sum(eval_fps_list) / len(eval_fps_list)
            bpp = sum(bpp_list) / len(bpp_list)
            best_psnr_value = sum(best_psnr_value_list) / len(best_psnr_value_list)
            best_ms_ssim_value = sum(best_ms_ssim_value_list) / len(best_ms_ssim_value_list)
            best_bpp = sum(best_bpp_list) / len(best_bpp_list)
        return psnr_value, ms_ssim_value, end_time, test_end_time, eval_fps, bpp, best_psnr_value, best_ms_ssim_value, best_bpp

    def test_GV3D2D(self, gaussian_model, gt_image, best=False):
        gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model.forward_quantize()
        mse_loss = F.mse_loss(out["render"].float(), gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        
        render_tensor = out["render"].float()
        num_time_steps = render_tensor.size(-1)
        
        ms_ssim_value = 0.0
        try:
            ms_ssim_values = []
            for t in range(num_time_steps):
                frame = render_tensor[..., t]
                gt_frame = gt_image[..., t]
                ms_ssim_values.append(
                    ms_ssim(frame, gt_frame, data_range=1, size_average=True).item()
                )
            ms_ssim_value = sum(ms_ssim_values) / len(ms_ssim_values)
        except AssertionError as e:
            self.logwriter.write("MS-SSIM could not be computed: " + str(e))
        
        m_bit, s_bit, r_bit, c_bit = out["unit_bit"]
        bpp = (m_bit + s_bit + r_bit + c_bit) / (self.H * self.W * self.T)
        tag = "Best Test" if best else "Test"
        self.logwriter.write(f"{tag} PSNR:{psnr:.4f}, MS_SSIM:{ms_ssim_value:.6f}, bpp:{bpp:.4f}")

        if self.save_imgs:
            transform = transforms.ToPILImage()
            label = "codec_best" if best else "codec" 

            for t in range(num_time_steps):
                img = render_tensor[0, :, :, :, t]  
                pil_image = transform(img)  
                name = f"{self.video_name}_fitting_t{t}_layer{self.layer}_{label}.png"
                pil_image.save(str(self.log_dir / name))
        return psnr, ms_ssim_value, bpp

    def test_GVGI(self, gaussian_model, gt_image, t=0, best=False):
        gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model.forward_quantize()
        mse_loss = F.mse_loss(out["render"].float(), gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        m_bit, s_bit, r_bit, c_bit = out["unit_bit"]
        bpp = (m_bit + s_bit + r_bit + c_bit)/self.H/self.W

        strings = "Best Test" if best else "Test"
        self.logwriter.write("{} PSNR:{:.4f}, MS_SSIM:{:.6f}, bpp:{:.4f}".format(strings, psnr, 
            ms_ssim_value, bpp))

        if self.save_imgs:
            transform = transforms.ToPILImage()
            img = transform(out["render"].float().squeeze(0))
            label = "codec_best" if best else "codec" 
            name = f"{self.video_name}_fitting_t{t}_layer{self.layer}_{label}.png"
            Path(self.log_dir / f'frame_{t+1:04}').mkdir(parents=True, exist_ok=True)
            img.save(str(self.log_dir / f'frame_{t+1:04}' / name))
        return psnr, ms_ssim_value, bpp

    def test(self, best=False):
        if self.model_name == "GV3D2D":
            psnr_value, ms_ssim_value, bpp = self.test_GV3D2D(self.gaussian_model, self.gt_image, best)
        elif self.model_name == "GVGI":
            psnr_value_list = []
            ms_ssim_value_list = []
            bpp_list = []
            for t in range(self.T):
                psnr_value, ms_ssim_value, bpp = self.test_GVGI(self.gaussian_model_list[t], self.gt_image[...,t], t, best)
                psnr_value_list.append(psnr_value)
                ms_ssim_value_list.append(ms_ssim_value)
                bpp_list.append(bpp)
            psnr_value = sum(psnr_value_list) / len(psnr_value_list)
            ms_ssim_value = sum(ms_ssim_value_list) / len(ms_ssim_value_list)
            bpp = sum(bpp_list) / len(bpp_list)
        return psnr_value, ms_ssim_value, bpp

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train quantized GaussianVideo3D2D model.")
    
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
    # Cache the args as a text string to save them in the output dir later
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
    image_length, start = args.num_frames, args.start_frame

    images_paths = []
    for i in range(start, start+image_length):
        image_path = Path(args.dataset) / f'frame_{i+1:04}.png'
        images_paths.append(image_path)

    trainer = GaussianVideo3D2DTrainerQuantize(layer=args.layer, images_paths=images_paths, model_name=args.model_name, args=args, num_frames=args.num_frames, start_frame=args.start_frame, video_name=args.data_name, log_dir=log_dir)
    psnr, ms_ssim, training_time, eval_time, eval_fps, bpp, best_psnr, best_ms_ssim, best_bpp = trainer.train()
    
    logwriter.write("Final Results - PSNR:{:.4f}, MS-SSIM:{:.4f}, bpp:{:.4f}, Best PSNR:{:.4f}, Best MS-SSIM:{:.4f}, Best bpp:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
            psnr, ms_ssim, bpp, best_psnr, best_ms_ssim, best_bpp, training_time, eval_time, eval_fps))


if __name__ == "__main__":
    main(sys.argv[1:])
