"""
Test script: GaussianVideo3D2D with layer 0 disabled and only a GaussianImage
checkpoint loaded for layer 1. Temporal extent is set so that t=2 (0-indexed)
holds the layer 1 content while self.T=3. Expected output: 2 black frames and
the GaussianImage rendered on the 2nd frame (index 2, i.e. 3rd frame).
"""
import os
import sys
import argparse
import tempfile
from pathlib import Path

import torch
import torchvision.transforms as transforms

from gaussianvideo3D2D import GaussianVideo3D2D


def create_empty_layer0_checkpoint(device, path):
    """Create a minimal layer 0 checkpoint with 0 gaussians (disables layer 0)."""
    state = {
        "_xyz_3D": torch.empty(0, 3, device=device),
        "_cholesky_3D": torch.empty(0, 6, device=device),
        "_features_dc_3D": torch.empty(0, 3, device=device),
        "_opacity_3D": torch.empty(0, 1, device=device),
    }
    torch.save(state, path)
    return path


def gaussian_image_ckpt_to_layer1_ckpt(gaussian_image_path, device, T, content_frame_idx, out_path):
    """
    Convert a GaussianImage (GaussianImage_Cholesky) checkpoint to 3D2D layer 1 format.
    Puts all gaussians on frame content_frame_idx; other frames get 0 gaussians.
    """
    ckpt = torch.load(gaussian_image_path, map_location=device)
    # GaussianImage uses _xyz (N, 2), _cholesky (N, 3), _features_dc, _opacity
    if "_xyz" not in ckpt:
        raise KeyError(
            "GaussianImage checkpoint must contain _xyz, _cholesky, _features_dc, _opacity. "
            "Got keys: " + str(list(ckpt.keys()))
        )
    N = ckpt["_xyz"].shape[0]
    # num_points_list: one (start, end) per frame. Only content_frame_idx has gaussians.
    num_points_list = []
    for t in range(T):
        if t == content_frame_idx:
            num_points_list.append((0, N))
        else:
            num_points_list.append((0, 0))
    # _opacity_3D for layer 0; we have 0 layer 0 gaussians
    opacity_3D = torch.empty(0, 1, device=device)
    state = {
        "_xyz_2D": ckpt["_xyz"].to(device),
        "_cholesky_2D": ckpt["_cholesky"].to(device),
        "_features_dc_2D": ckpt["_features_dc"].to(device),
        "_opacity_2D": ckpt["_opacity"].to(device),
        "_opacity_3D": opacity_3D,
        "gaussian_num_list": num_points_list,
    }
    torch.save(state, out_path)
    return out_path


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Test 3D2D with layer 0 disabled and GaussianImage as layer 1 on frame t=2."
    )
    parser.add_argument("--gaussian_image_ckpt", type=str, required=True,
                        help="Path to GaussianImage checkpoint (.pth.tar)")
    parser.add_argument("--output_dir", type=str, default="./test_3D2D_out",
                        help="Directory to save output frames")
    parser.add_argument("--H", type=int, default=256, help="Image height")
    parser.add_argument("--W", type=int, default=256, help="Image width")
    parser.add_argument("--T", type=int, default=3, help="Number of temporal frames")
    parser.add_argument("--content_frame", type=int, default=2,
                        help="Frame index (0-based) where GaussianImage is shown; others are black")
    parser.add_argument("--num_points", type=int, default=50000, help="Num points (for model init)")
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-2)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    H, W, T = args.H, args.W, args.T
    BLOCK_H, BLOCK_W, BLOCK_T = 16, 16, 1

    # Create temp checkpoints: empty layer 0 and layer 1 from GaussianImage
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        layer0_path = tmpdir / "layer0_empty.pth.tar"
        layer1_path = tmpdir / "layer1_from_gi.pth.tar"

        create_empty_layer0_checkpoint(device, layer0_path)
        gaussian_image_ckpt_to_layer1_ckpt(
            args.gaussian_image_ckpt, device, T, args.content_frame, layer1_path
        )

        kwargs = {
            "layer": 1,
            "loss_type": "L2",
            "opt_type": "adan",
            "H": H,
            "W": W,
            "T": T,
            "BLOCK_H": BLOCK_H,
            "BLOCK_W": BLOCK_W,
            "BLOCK_T": BLOCK_T,
            "device": device,
            "quantize": False,
            "num_points": args.num_points,
            "iterations": args.iterations,
            "lr": args.lr,
        }
        model = GaussianVideo3D2D(**kwargs)
        model._create_data_from_checkpoint(str(layer0_path), str(layer1_path))
        # Layer 0 was loaded from empty ckpt (0 gaussians); set for any code that uses it
        model.num_points_layer0 = model._xyz_3D.shape[0]
        model.to(device)
        model.eval()

        with torch.no_grad():
            out = model.forward()
        render = out["render"]  # [1, C, H, W, T]

    os.makedirs(args.output_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()
    for t in range(T):
        frame = render[0, :, :, :, t].clamp(0, 1)
        pil_img = to_pil(frame.cpu())
        path = Path(args.output_dir) / f"frame_{t:04d}.png"
        pil_img.save(str(path))
        print(f"Saved {path} (frame t={t}, expect {'GaussianImage' if t == args.content_frame else 'black'})")

    print("Done. Expected: 2 black images and 1 GaussianImage output on frame index 2.")


if __name__ == "__main__":
    main(sys.argv[1:])
