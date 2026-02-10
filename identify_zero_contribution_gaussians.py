"""
Identify Gaussians that contribute to no pixels (always skipped in rasterize due to
sigma < 0 or alpha < 1/255 at every tile they intersect). Prints their projected xyz,
L elements, covariance, and conic in detail.

Requires the CUDA change that records gaussian_contributed in forward.cu (rasterize kernel).

Usage:
  python identify_zero_contribution_gaussians.py --checkpoint path/to/gaussian_model.pth.tar --H 1080 --W 1920 --T 50 [--out_file zero_contrib.txt]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from gsplat.project_gaussians_video import project_gaussians_video
from gsplat.rasterize_sum_video import rasterize_gaussians_sum_video

CHOLESKY_BOUND = torch.tensor([0.5, 0, 0, 0.5, 0, 0.5], dtype=torch.float32)


def load_checkpoint(path: str, device="cpu"):
    """Load checkpoint; support GaussianVideo and 3D2D formats."""
    ckpt = torch.load(path, map_location=device)
    if "_xyz_3D" in ckpt:
        cholesky_raw = ckpt["_cholesky_3D"]
        xyz_raw = ckpt["_xyz_3D"]
        features_dc = ckpt["_features_dc_3D"]
        opacity_raw = ckpt["_opacity_3D"]
    elif "_xyz" in ckpt:
        cholesky_raw = ckpt["_cholesky"]
        xyz_raw = ckpt["_xyz"]
        features_dc = ckpt["_features_dc"]
        opacity_raw = ckpt["_opacity"]
    else:
        raise KeyError("Checkpoint must contain _xyz/_cholesky or _xyz_3D/_cholesky_3D")
    bound = CHOLESKY_BOUND.to(cholesky_raw.device).view(1, 6)
    cholesky_elements = cholesky_raw + bound
    return cholesky_elements, xyz_raw, features_dc, opacity_raw


def cholesky_to_covariance(L_flat: torch.Tensor) -> torch.Tensor:
    """(N, 6) L -> (N, 6) cov [Cxx, Cxy, Cxz, Cyy, Cyz, Czz]."""
    l11, l21, l31 = L_flat[:, 0], L_flat[:, 1], L_flat[:, 2]
    l22, l32, l33 = L_flat[:, 3], L_flat[:, 4], L_flat[:, 5]
    return torch.stack([
        l11 * l11, l11 * l21, l11 * l31,
        l21 * l21 + l22 * l22, l21 * l31 + l22 * l32,
        l31 * l31 + l32 * l32 + l33 * l33
    ], dim=1)


def main():
    parser = argparse.ArgumentParser(description="Identify Gaussians with zero pixel contribution")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--H", type=int, default=1080)
    parser.add_argument("--W", type=int, default=1920)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_file", type=str, default=None, help="Optional file to write details")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Loading checkpoint...")
    cholesky_elements, xyz_raw, features_dc, opacity_raw = load_checkpoint(args.checkpoint, device)
    xyz = torch.tanh(xyz_raw).to(device)
    cholesky_elements = cholesky_elements.to(device)
    cov_flat = cholesky_to_covariance(cholesky_elements)
    N = xyz.shape[0]

    BLOCK_W, BLOCK_H, BLOCK_T = 16, 16, 1
    tile_bounds = (
        (args.W + BLOCK_W - 1) // BLOCK_W,
        (args.H + BLOCK_H - 1) // BLOCK_H,
        (args.T + BLOCK_T - 1) // BLOCK_T,
    )

    print("Running project_gaussians_video...")
    xys, depths, radii, conics, num_tiles_hit = project_gaussians_video(
        xyz.float().contiguous(),
        cholesky_elements.float().contiguous(),
        args.H, args.W, args.T, tile_bounds,
    )

    colors = torch.sigmoid(features_dc).float().to(device).contiguous()
    opacity = torch.sigmoid(opacity_raw).to(device).contiguous()
    background = torch.ones(3, dtype=torch.float32, device=device)

    print("Running rasterize with return_contribution=True...")
    out_img, final_Ts, final_idx, gaussian_contributed = rasterize_gaussians_sum_video(
        xys, depths, radii, conics, num_tiles_hit,
        colors, opacity, args.H, args.W, args.T,
        BLOCK_H, BLOCK_W, BLOCK_T,
        background=background, return_alpha=False, return_contribution=True,
    )

    contrib = gaussian_contributed.cpu().numpy()
    zero_mask = (contrib == 0)
    zero_indices = np.where(zero_mask)[0].tolist()
    num_zero = len(zero_indices)

    lines = [
        f"Checkpoint: {args.checkpoint}",
        f"H={args.H} W={args.W} T={args.T}",
        f"Total Gaussians: {N}",
        f"Zero-contribution Gaussians (sigma<0 or alpha<1/255 at every pixel): {num_zero}",
        "",
    ]

    if num_zero == 0:
        print("\n".join(lines))
        print("No zero-contribution Gaussians found.")
        if args.out_file:
            Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_file, "w") as f:
                f.write("\n".join(lines))
        return

    xys_np = xys.cpu().numpy()
    L_np = cholesky_elements.cpu().numpy()
    cov_np = cov_flat.cpu().numpy()
    conics_np = conics.cpu().numpy()
    radii_np = radii.cpu().numpy()
    num_tiles_np = num_tiles_hit.cpu().numpy()

    for idx in zero_indices:
        block = [
            f"============ Gaussian index {idx} (zero contribution) ============",
            f"  Projected xyz (pixel space, from project_gaussians_video):",
            f"    x = {xys_np[idx, 0]:.6f},  y = {xys_np[idx, 1]:.6f},  z(t) = {xys_np[idx, 2]:.6f}",
            f"  Radius (from projection): {radii_np[idx]}",
            f"  Num tiles hit: {num_tiles_np[idx]}",
            f"  L elements (cholesky, 6): [l11, l21, l31, l22, l32, l33]",
            f"    {L_np[idx, 0]:.6f}, {L_np[idx, 1]:.6f}, {L_np[idx, 2]:.6f}, {L_np[idx, 3]:.6f}, {L_np[idx, 4]:.6f}, {L_np[idx, 5]:.6f}",
            f"  Covariance (6): [Cxx, Cxy, Cxz, Cyy, Cyz, Czz]",
            f"    {cov_np[idx, 0]:.6f}, {cov_np[idx, 1]:.6f}, {cov_np[idx, 2]:.6f}, {cov_np[idx, 3]:.6f}, {cov_np[idx, 4]:.6f}, {cov_np[idx, 5]:.6f}",
            f"  Conic (6, from projection): [invCxx, invCxy, invCxz, invCyy, invCyz, invCzz]",
            f"    {conics_np[idx, 0]:.6f}, {conics_np[idx, 1]:.6f}, {conics_np[idx, 2]:.6f}, {conics_np[idx, 3]:.6f}, {conics_np[idx, 4]:.6f}, {conics_np[idx, 5]:.6f}",
            "",
        ]
        lines.extend(block)

    text = "\n".join(lines)
    print(text)

    if args.out_file:
        Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_file, "w") as f:
            f.write(text)
        print(f"Written to {args.out_file}")


if __name__ == "__main__":
    main()
