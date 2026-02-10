"""
Identify Gaussians that contribute to no pixels (always skipped in rasterize due to
sigma < 0 or alpha < 1/255 at every tile they intersect). Prints their projected xyz,
L elements, covariance, and conic in detail.

Requires the CUDA change that records gaussian_contributed in forward.cu (rasterize kernel).

Usage:
  python identify_zero_contribution_gaussians.py --checkpoint path/to/gaussian_model.pth.tar --H 1080 --W 1920 --T 50 [--out_file zero_contrib.txt]
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    parser.add_argument("--detail_zero_intersection", action="store_true", help="Print one-line details for each zero-intersection Gaussian (if count <= 200)")
    parser.add_argument("--gt_dir", type=str, default=None, help="Directory of ground-truth frames (frame_0001.png, frame_0002.png, ...) to compute PSNR and verify rendering")
    parser.add_argument("--start_frame", type=int, default=1, help="Start frame index for GT (1 = frame_0001.png)")
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

    num_tiles_np = num_tiles_hit.cpu().numpy()
    zero_intersection_mask = (num_tiles_np == 0)
    zero_intersection_indices = np.where(zero_intersection_mask)[0].tolist()
    num_zero_intersection = len(zero_intersection_indices)

    colors = features_dc.float().to(device).contiguous()
    opacity = torch.sigmoid(opacity_raw).to(device).contiguous()
    background = torch.ones(3, dtype=torch.float32, device=device)

    print("Running rasterize with return_contribution=True...")
    out_img, final_Ts, final_idx, gaussian_contributed = rasterize_gaussians_sum_video(
        xys, depths, radii, conics, num_tiles_hit,
        colors, opacity, args.H, args.W, args.T,
        BLOCK_H, BLOCK_W, BLOCK_T,
        background=background, return_alpha=False, return_contribution=True,
    )
    # out_img: [T, H, W, 3] -> convert to [1, 3, H, W, T] to match GT format (same as GaussianVideo.forward)
    render = torch.clamp(out_img, 0, 1).float()
    render = render.view(-1, args.T, args.H, args.W, 3).permute(0, 4, 2, 3, 1).contiguous()

    contrib = gaussian_contributed.cpu().numpy()
    zero_contrib_mask = (contrib == 0)
    zero_indices = np.where(zero_contrib_mask)[0].tolist()
    num_zero = len(zero_indices)

    xys_np = xys.cpu().numpy()
    L_np = cholesky_elements.cpu().numpy()
    cov_np = cov_flat.cpu().numpy()
    conics_np = conics.cpu().numpy()
    radii_np = radii.cpu().numpy()

    psnr_line = ""
    if args.gt_dir is not None:
        gt_dir = Path(args.gt_dir)
        from utils import images_paths_to_tensor
        gt_paths = [gt_dir / f"frame_{args.start_frame + i:04d}.png" for i in range(args.T)]
        if not all(p.exists() for p in gt_paths):
            psnr_line = f"  PSNR vs GT: skipped (missing frames in {args.gt_dir}; expected frame_{args.start_frame:04d}.png ... frame_{args.start_frame + args.T - 1:04d}.png)"
        else:
            gt_tensor = images_paths_to_tensor(gt_paths).to(device)
            if gt_tensor.shape[2] != args.H or gt_tensor.shape[3] != args.W or gt_tensor.shape[4] != args.T:
                psnr_line = f"  PSNR vs GT: skipped (GT shape [1,{gt_tensor.shape[1]},{gt_tensor.shape[2]},{gt_tensor.shape[3]},{gt_tensor.shape[4]}] != render [1,3,{args.H},{args.W},{args.T}])"
            else:
                mse = F.mse_loss(render.float(), gt_tensor.float()).item()
                psnr = 10.0 * math.log10(1.0 / (mse + 1e-8))
                psnr_line = f"  PSNR vs GT ({args.gt_dir}): {psnr:.4f} dB  (MSE={mse:.6f})"

    lines = [
        f"Checkpoint: {args.checkpoint}",
        f"H={args.H} W={args.W} T={args.T}",
        f"Total Gaussians: {N}",
        "",
        "--- Render vs ground truth ---",
        psnr_line if psnr_line else "  PSNR vs GT: not computed (pass --gt_dir <dir> with frame_0001.png, ...)",
        "",
        "--- Zero intersection (num_tiles_hit == 0) ---",
        f"  These Gaussians intersect NO tile in the volume (radius 0 or bbox outside). They are never considered in rasterize.",
        f"  Count: {num_zero_intersection}",
        "",
        "--- Zero contribution (contributed to no pixel) ---",
        f"  These Gaussians contributed to NO pixel (sigma<0 or alpha<1/255 at every voxel where they were tested).",
        f"  Count: {num_zero}",
        f"  (Zero-intersection Gaussians are a subset of zero-contribution.)",
        "",
    ]

    if num_zero_intersection > 0:
        lines.append("  Zero-intersection Gaussian indices (first 50):")
        lines.append("    " + ", ".join(str(i) for i in zero_intersection_indices[:50]))
        if num_zero_intersection > 50:
            lines.append(f"    ... and {num_zero_intersection - 50} more.")
        if args.detail_zero_intersection and num_zero_intersection <= 200:
            lines.append("")
            lines.append("  Details for zero-intersection Gaussians:")
            for idx in zero_intersection_indices:
                lines.append(f"  --- index {idx}: xyz=({xys_np[idx,0]:.4f},{xys_np[idx,1]:.4f},{xys_np[idx,2]:.4f}) radius={radii_np[idx]} L=[{L_np[idx,0]:.4f},{L_np[idx,1]:.4f},{L_np[idx,2]:.4f},{L_np[idx,3]:.4f},{L_np[idx,4]:.4f},{L_np[idx,5]:.4f}]")
        lines.append("")

    if num_zero == 0:
        print("\n".join(lines))
        print("No zero-contribution Gaussians found.")
        if args.out_file:
            Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_file, "w") as f:
                f.write("\n".join(lines))
        return

    for idx in zero_indices:
        no_tiles = num_tiles_np[idx] == 0
        block = [
            f"============ Gaussian index {idx} (zero contribution" + (", 0 intersection" if no_tiles else "") + ") ============",
            f"  Projected xyz (pixel space, from project_gaussians_video):",
            f"    x = {xys_np[idx, 0]:.6f},  y = {xys_np[idx, 1]:.6f},  z(t) = {xys_np[idx, 2]:.6f}",
            f"  Radius (from projection): {radii_np[idx]}",
            f"  Num tiles hit: {num_tiles_np[idx]}" + (" (0 = never considered in rasterize)" if no_tiles else ""),
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
