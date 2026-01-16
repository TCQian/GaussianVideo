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
from analyze_quantizers import analyze_quantizer_similarity, plot_quantizer_distributions
class GaussianVideo3D2DTrainerQuantize:
    """Tests quantized GaussianVideo models on a video (i.e. a set of frames)."""
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

        self.device = torch.device("cuda:0")
        self.gt_video = images_paths_to_tensor(images_paths).to(self.device)  # [1, C, H, W, T]
        self.video_name = video_name
        self.model_name = model_name
        self.layer = layer
        assert self.layer == 1, "Only layer 1 is supported for testing"

        # Determine spatial and temporal dimensions
        self.H, self.W, self.T = self.gt_video.shape[2], self.gt_video.shape[3], self.gt_video.shape[4]
        BLOCK_H, BLOCK_W, BLOCK_T = 16, 16, 1  # adjust BLOCK_T if needed
        self.iterations = args.iterations
        self.iterations = args.iterations
        self.num_points = args.num_points
        self.save_imgs = args.save_imgs
        self.log_dir = log_dir

        self.gaussian_model_layer0 = GaussianVideo3D2D(
                layer=0,
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
            )
        self.gaussian_model_layer0._create_data_from_checkpoint(args.model_path_layer0, None)
        self.gaussian_model_layer0.to(self.device)

        if self.model_name == "GV3D2D":
            self.gaussian_model = GaussianVideo3D2D(
                layer=1,
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
            )
            self.gaussian_model._create_data_from_checkpoint(args.model_path_layer0, args.model_path_layer1)
            self.gaussian_model.to(self.device)
            self.gaussian_model.create_en_decoded_layer0() # initialize decoded layer 0 for layer 1 training

        if self.model_name == "GVGI":
            assert self.layer == 1, "GVGI is only able to process Layer 1 "
            checkpoint_layer0 = torch.load(args.model_path_layer0, map_location=self.device)
            self.init_num_points_layer1 = int((self.num_points * self.T) - checkpoint_layer0['_xyz_3D'].shape[0])
            num_points_per_frame = int(self.init_num_points_layer1 / self.T)
            print(f"GVGI: Available number of gaussians: {self.init_num_points_layer1} for layer 1")

            # get the background image tensor of layer 0
            self.gaussian_model_layer0.eval()
            with torch.no_grad():
                encoding_dict = self.gaussian_model_layer0.compress_wo_ec()
                out = self.gaussian_model_layer0.decompress_wo_ec(encoding_dict)
                bg_tensor = out["render"].float()

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

                gaussian_model = GaussianImage_Cholesky(
                    background_image=bg_tensor[0, :, :, :, t].squeeze(0).permute(1, 2, 0),
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
                    gaussian_model.load_state_dict(checkpoint, strict=False)
                    print(f"Loaded checkpoint from: {checkpoint_file_path}")

                self.gaussian_model_list.append(gaussian_model)

        self.logwriter = LogWriter(self.log_dir, train=False)

    def test_GV3D2D(self, gaussian_model, gt_image, layer=0):
        gaussian_model.eval()
        with torch.no_grad():
            encoding_dict = gaussian_model.compress_wo_ec()
            out = gaussian_model.decompress_wo_ec(encoding_dict)
            start_time = time.time()
            for i in range(100):
                _ = gaussian_model.decompress_wo_ec(encoding_dict)
            end_time = (time.time() - start_time) / 100
        data_dict = gaussian_model.analysis_wo_ec(encoding_dict)

        # Perform quantizer similarity analysis (only if quantization is enabled)
        try:
            if not gaussian_model.quantize:
                print(f"Skipping quantizer analysis for layer {layer}: quantization not enabled")
            else:
                print(f"\n{'='*80}")
                print(f"Quantizer Similarity Analysis - Layer {layer}")
                print(f"{'='*80}")
                print(f"Log directory: {self.log_dir}")
                
                quantizer_results = analyze_quantizer_similarity(gaussian_model, layer=layer)
                
                if quantizer_results is None:
                    print(f"Warning: analyze_quantizer_similarity returned None for layer {layer}")
                else:
                    # Save analysis results (convert to serializable format)
                    def convert_to_serializable(obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, (np.integer, np.floating)):
                            return float(obj) if isinstance(obj, np.floating) else int(obj)
                        elif isinstance(obj, dict):
                            return {k: convert_to_serializable(v) for k, v in obj.items()}
                        elif isinstance(obj, (list, tuple)):
                            return [convert_to_serializable(item) for item in obj]
                        elif isinstance(obj, torch.Tensor):
                            return obj.cpu().numpy().tolist()
                        return obj
                    
                    serializable_results = convert_to_serializable(quantizer_results)
                    analysis_file = self.log_dir / f"quantizer_analysis_layer_{layer}.npy"
                    
                    # Ensure directory exists
                    analysis_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    print(f"Saving analysis results to: {analysis_file}")
                    print(f"  File exists before save: {analysis_file.exists()}")
                    print(f"  Directory exists: {analysis_file.parent.exists()}")
                    print(f"  Directory is writable: {os.access(analysis_file.parent, os.W_OK)}")
                    
                    try:
                        np.save(str(analysis_file), serializable_results)
                        print(f"  np.save completed")
                    except Exception as save_error:
                        print(f"  ERROR during np.save: {save_error}")
                        raise
                    
                    # Verify file was created
                    if analysis_file.exists():
                        file_size = analysis_file.stat().st_size
                        print(f"✓ Analysis file saved successfully: {analysis_file} ({file_size} bytes)")
                    else:
                        print(f"✗ ERROR: Analysis file was not created: {analysis_file}")
                        print(f"  Current working directory: {os.getcwd()}")
                        print(f"  Absolute path: {analysis_file.absolute()}")
                    
                    # Create visualizations
                    try:
                        plot_file = self.log_dir / f"quantizer_distributions_layer_{layer}.png"
                        
                        # Ensure directory exists
                        plot_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        print(f"Creating visualization: {plot_file}")
                        print(f"  File exists before save: {plot_file.exists()}")
                        print(f"  Directory exists: {plot_file.parent.exists()}")
                        
                        plot_quantizer_distributions(quantizer_results, output_path=str(plot_file))
                        
                        # Verify plot was created
                        if plot_file.exists():
                            file_size = plot_file.stat().st_size
                            print(f"✓ Visualization saved successfully: {plot_file} ({file_size} bytes)")
                        else:
                            print(f"✗ ERROR: Visualization file was not created: {plot_file}")
                            print(f"  Current working directory: {os.getcwd()}")
                            print(f"  Absolute path: {plot_file.absolute()}")
                    except Exception as e:
                        print(f"ERROR: Could not create quantizer visualizations: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Add key metrics to data_dict for logging
                    if 'cholesky' in quantizer_results:
                        cholesky_util = quantizer_results['cholesky']['utilization']
                        data_dict['cholesky_utilization'] = cholesky_util
                        data_dict['cholesky_compression_ratio'] = quantizer_results['cholesky']['compression_ratio']
                        self.logwriter.write(f"Layer {layer}: Cholesky utilization: {cholesky_util*100:.1f}%, compression ratio: {quantizer_results['cholesky']['compression_ratio']:.2f}x")
                    
                    if 'features' in quantizer_results:
                        features_util = quantizer_results['features']['utilization']
                        data_dict['features_utilization'] = features_util
                        data_dict['features_compression_ratio'] = quantizer_results['features']['compression_ratio']
                        optimal_codebook = max([s['unique_count'] for s in quantizer_results['features']['per_quantizer_stats']])
                        current_codebook = quantizer_results['features']['codebook_size']
                        self.logwriter.write(f"Layer {layer}: Features utilization: {features_util*100:.1f}%, compression ratio: {quantizer_results['features']['compression_ratio']:.2f}x")
                        self.logwriter.write(f"Layer {layer}: Features codebook usage: {optimal_codebook}/{current_codebook} entries")
        except Exception as e:
            print(f"ERROR: Quantizer analysis failed for layer {layer}: {e}")
            import traceback
            traceback.print_exc()

        out_video = out["render"].float()  # Expected shape: [1, C, H, W, T]
        mse_loss = F.mse_loss(out_video, gt_image)
        psnr = 10 * math.log10(1.0 / (mse_loss.item() + 1e-8))
        
        T = out_video.shape[-1]
        ms_ssim_total = 0.0
        for t in range(T):
            frame_pred = out_video[..., t]  # shape: [1, C, H, W]
            frame_gt = gt_image[..., t]  # shape: [1, C, H, W]
            ms_ssim_total += ms_ssim(frame_pred, frame_gt, data_range=1, size_average=True).item()
        
            if self.save_imgs:
                transform = transforms.ToPILImage()
                img = transform(frame_pred.squeeze(0))
                name = f"{self.video_name}_fitting_t{t}_layer{layer}_codec_best.png"
                img.save(str(self.log_dir / name))

        ms_ssim_value = ms_ssim_total / T
        FPS = self.T/end_time
        data_dict["psnr"] = psnr
        data_dict["ms-ssim"] = ms_ssim_value
        data_dict["rendering_time"] = end_time
        data_dict["rendering_fps"] = FPS

        np.save(self.log_dir / f"test_layer_{layer}.npy", data_dict)
        self.logwriter.write("Layer {}: Eval time:{:.8f}s, FPS:{:.4f}".format(layer, end_time, FPS))
        self.logwriter.write("Layer {}: PSNR:{:.4f}, MS_SSIM:{:.6f}, bpp:{:.4f}".format(layer, psnr, ms_ssim_value, data_dict["bpp"]))
        self.logwriter.write("Layer {}: position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}".format(
            layer, data_dict["position_bpp"], data_dict["cholesky_bpp"], data_dict["feature_dc_bpp"]))
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

        # Perform quantizer similarity analysis for GVGI (GaussianImage_Cholesky)
        try:
            if not gaussian_model.quantize:
                print(f"Skipping quantizer analysis for frame {t}: quantization not enabled")
            else:
                print(f"\n{'='*80}")
                print(f"Quantizer Similarity Analysis - GVGI Frame {t}")
                print(f"{'='*80}")
                
                # Analyze cholesky quantizer
                if hasattr(gaussian_model, 'cholesky_quantizer') and gaussian_model.cholesky_quantizer is not None:
                    cholesky_results = gaussian_model.cholesky_quantizer.analyze(gaussian_model._cholesky, verbose=True)
                    data_dict['cholesky_utilization'] = cholesky_results['utilization']
                    data_dict['cholesky_compression_ratio'] = cholesky_results['compression_ratio']
                    self.logwriter.write(f"Frame {t}: Cholesky utilization: {cholesky_results['utilization']*100:.1f}%, compression ratio: {cholesky_results['compression_ratio']:.2f}x")
                
                # Analyze features quantizer
                if hasattr(gaussian_model, 'features_dc_quantizer') and gaussian_model.features_dc_quantizer is not None:
                    features_results = gaussian_model.features_dc_quantizer.analyze(gaussian_model._features_dc, verbose=True)
                    data_dict['features_utilization'] = features_results['utilization']
                    data_dict['features_compression_ratio'] = features_results['compression_ratio']
                    optimal_codebook = max([s['unique_count'] for s in features_results['per_quantizer_stats']])
                    current_codebook = features_results['codebook_size']
                    self.logwriter.write(f"Frame {t}: Features utilization: {features_results['utilization']*100:.1f}%, compression ratio: {features_results['compression_ratio']:.2f}x")
                    self.logwriter.write(f"Frame {t}: Features codebook usage: {optimal_codebook}/{current_codebook} entries")
        except Exception as e:
            print(f"Warning: Quantizer analysis failed for frame {t}: {e}")
            import traceback
            traceback.print_exc()
    
        out_img = out["render"].float()
        mse_loss = F.mse_loss(out_img, gt_image)
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out_img, gt_image, data_range=1, size_average=True).item()
        
        if self.save_imgs:
            transform = transforms.ToPILImage()
            img = transform(out_img.squeeze(0))
            name = f"{self.video_name}_fitting_t{t}_layer{self.layer}_codec_best.png"
            img.save(str(self.log_dir / name))

        data_dict["psnr"] = psnr
        data_dict["ms-ssim"] = ms_ssim_value
        data_dict["rendering_time"] = end_time
        data_dict["rendering_fps"] = 1/end_time
        np.save(self.log_dir / f"test_layer_{self.layer}.npy", data_dict)
        self.logwriter.write("Layer 1:Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, 1/end_time))
        self.logwriter.write("Layer 1: PSNR:{:.4f}, MS_SSIM:{:.6f}, bpp:{:.4f}".format(psnr, ms_ssim_value, data_dict["bpp"]))
        self.logwriter.write("Layer 1: position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}".format(data_dict["position_bpp"], data_dict["cholesky_bpp"], data_dict["feature_dc_bpp"]))
        return data_dict

    def test(self):
        data_dict = self.test_GV3D2D(self.gaussian_model_layer0, self.gt_video, layer=0)

        if self.model_name == "GV3D2D":
            data_dict_layer1 = self.test_GV3D2D(self.gaussian_model, self.gt_video, layer=1)
            data_dict["psnr_layer1"] = data_dict_layer1["psnr"]
            data_dict["ms-ssim_layer1"] = data_dict_layer1["ms-ssim"]
            data_dict["rendering_time_layer1"] = data_dict_layer1["rendering_time"]
            data_dict["rendering_fps_layer1"] = data_dict_layer1["rendering_fps"]
            data_dict["bpp_layer1"] = data_dict_layer1["bpp"]
            data_dict["position_bpp_layer1"] = data_dict_layer1["position_bpp"]
            data_dict["cholesky_bpp_layer1"] = data_dict_layer1["cholesky_bpp"]
            data_dict["feature_dc_bpp_layer1"] = data_dict_layer1["feature_dc_bpp"]
        elif self.model_name == "GVGI":
            psnr_list = []
            ms_ssim_list = []
            rendering_time_list = []
            rendering_fps_list = []
            bpp_list = []
            position_bpp_list = []
            cholesky_bpp_list = []
            feature_dc_bpp_list = []
            for t in range(self.T):
                data_dict_layer1 = self.test_GVGI(self.gaussian_model_list[t], self.gt_video[..., t], t)
                psnr_list.append(data_dict_layer1["psnr"])
                ms_ssim_list.append(data_dict_layer1["ms-ssim"])
                rendering_time_list.append(data_dict_layer1["rendering_time"])
                rendering_fps_list.append(data_dict_layer1["rendering_fps"])
                bpp_list.append(data_dict_layer1["bpp"])
                position_bpp_list.append(data_dict_layer1["position_bpp"])
                cholesky_bpp_list.append(data_dict_layer1["cholesky_bpp"])
                feature_dc_bpp_list.append(data_dict_layer1["feature_dc_bpp"])

            data_dict["psnr_layer1"] = sum(psnr_list) / len(psnr_list)
            data_dict["ms-ssim_layer1"] = sum(ms_ssim_list) / len(ms_ssim_list)
            data_dict["rendering_time_layer1"] = sum(rendering_time_list) / len(rendering_time_list)
            data_dict["rendering_fps_layer1"] = sum(rendering_fps_list) / len(rendering_fps_list)
            data_dict["bpp_layer1"] = sum(bpp_list) / len(bpp_list)
            data_dict["position_bpp_layer1"] = sum(position_bpp_list) / len(position_bpp_list)
            data_dict["cholesky_bpp_layer1"] = sum(cholesky_bpp_list) / len(cholesky_bpp_list)
            data_dict["feature_dc_bpp_layer1"] = sum(feature_dc_bpp_list) / len(feature_dc_bpp_list)
        
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
    
    log_dir = log_dir / 'test'
    os.makedirs(log_dir, exist_ok=True)
    logwriter = LogWriter(log_dir, train=False)

    images_paths = []
    for i in range(args.start_frame, args.start_frame + args.num_frames):
        image_path = Path(args.dataset) / f'frame_{i+1:04}.png'
        images_paths.append(image_path)

    trainer = GaussianVideo3D2DTrainerQuantize(layer=args.layer, images_paths=images_paths, model_name=args.model_name, args=args, num_frames=args.num_frames, start_frame=args.start_frame, video_name=args.data_name, log_dir=log_dir)
    
    data_dict = trainer.test()
    logwriter.write(f"Layer 0: {args.model_name}")
    logwriter.write("Video: {}x{}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, bpp:{:.4f}, Eval time:{:.8f}s, FPS:{:.4f}, position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}".format(
        trainer.H, trainer.W, trainer.T, data_dict["psnr"], data_dict["ms-ssim"], data_dict["bpp"],
        data_dict["rendering_time"], data_dict["rendering_fps"],
        data_dict["position_bpp"], data_dict["cholesky_bpp"], data_dict["feature_dc_bpp"]
    ))

    logwriter.write(f"Layer 1: {args.model_name}")
    logwriter.write("Video: {}x{}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, bpp:{:.4f}, Eval time:{:.8f}s, FPS:{:.4f}, position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}".format(
        trainer.H, trainer.W, trainer.T, data_dict["psnr_layer1"], data_dict["ms-ssim_layer1"], data_dict["bpp_layer1"],
        data_dict["rendering_time_layer1"], data_dict["rendering_fps_layer1"],
        data_dict["position_bpp_layer1"], data_dict["cholesky_bpp_layer1"], data_dict["feature_dc_bpp_layer1"]
    ))
if __name__ == "__main__":
    main(sys.argv[1:])
