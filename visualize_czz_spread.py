"""
Visualize single Gaussians in full T space: pick one with big Cxx/Cyy and one with
small Cxx/Cyy, then render each with original / doubled / halved Czz to see how
Czz affects 2D xy spread (tile_num, radius) and color spread per frame.

All Czz scaling is done in this script by scaling the Cholesky (l31, l32, l33).
No changes to the gsplat folder are required.

Usage:
  python visualize_czz_spread.py --checkpoint path/to/gaussian_model.pth.tar --H 1080 --W 1920 --T 50 --out_dir ./viz_czz
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Reuse same bound as GaussianVideo
CHOLESKY_BOUND = torch.tensor([0.5, 0, 0, 0.5, 0, 0.5], dtype=torch.float32)

# -----------------------------------------------------------------------------
# Czz variant factors: change here to get different temporal spread.
# Czz_new = CZZ_FACTOR * Czz_original. Affects tile_num and radius from projection.
# 1.0 = original, 2.0 = doubled Czz, 0.5 = halved Czz
# Implemented by scaling (l31, l32, l33) by sqrt(CZZ_FACTOR).
# -----------------------------------------------------------------------------
CZZ_VARIANTS = [
    1.0,   # original
    2.0,   # doubled Czz (more spread in T -> larger radius / more tiles)
    0.5,   # halved Czz (less spread in T -> smaller radius / fewer tiles)
]
CZZ_VARIANT_LABELS = ["original_czz", "doubled_czz", "halved_czz"]


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
    Cxx = l11 * l11
    Cxy = l11 * l21
    Cxz = l11 * l31
    Cyy = l21 * l21 + l22 * l22
    Cyz = l21 * l31 + l22 * l32
    Czz = l31 * l31 + l32 * l32 + l33 * l33
    return torch.stack([Cxx, Cxy, Cxz, Cyy, Cyz, Czz], dim=1)


def scale_cholesky_czz(cholesky_1x6: torch.Tensor, czz_factor: float) -> torch.Tensor:
    """
    Scale Czz by czz_factor by scaling (l31, l32, l33) by sqrt(czz_factor).
    Czz = l31^2 + l32^2 + l33^2, so new_Czz = czz_factor * Czz.
    """
    out = cholesky_1x6.clone()
    scale = float(np.sqrt(czz_factor))
    out[:, 2] *= scale   # l31
    out[:, 4] *= scale   # l32
    out[:, 5] *= scale   # l33
    return out


def pick_big_small_xy(cov_flat: torch.Tensor, top_pct: float = 15.0, bottom_pct: float = 15.0) -> tuple:
    """
    Pick from histogram distribution: big = has big Cxx OR big Cyy (top percentile),
    small = bottom percentile by xy spread.
    Returns (idx_big_xy, idx_small_xy).
    """
    Cxx = cov_flat[:, 0]
    Cyy = cov_flat[:, 3]
    valid = (Cxx > 1e-10) & (Cyy > 1e-10)
    if valid.sum() == 0:
        valid = torch.ones(cov_flat.shape[0], dtype=torch.bool, device=cov_flat.device)
    Cxx_v, Cyy_v = Cxx[valid], Cyy[valid]
    # Big: in top top_pct% by Cxx OR top top_pct% by Cyy (histogram upper tail)
    p_high = 100.0 - top_pct
    thresh_Cxx = torch.quantile(Cxx_v.float(), p_high / 100.0).item()
    thresh_Cyy = torch.quantile(Cyy_v.float(), p_high / 100.0).item()
    big_mask = valid & ((Cxx >= thresh_Cxx) | (Cyy >= thresh_Cyy))
    if big_mask.sum() == 0:
        big_mask = valid
    # Among big candidates, pick the one with largest (Cxx + Cyy) for visibility
    score_big = torch.where(big_mask, Cxx + Cyy, torch.zeros_like(Cxx))
    idx_big = score_big.argmax().item()

    # Small: bottom bottom_pct% by (Cxx + Cyy)
    sum_xy = Cxx + Cyy
    thresh_small = torch.quantile(sum_xy[valid].float(), bottom_pct / 100.0).item()
    small_mask = valid & (sum_xy <= thresh_small)
    if small_mask.sum() == 0:
        small_mask = valid
    idx_small = torch.where(small_mask, sum_xy, torch.full_like(sum_xy, float("inf"))).argmin().item()
    return idx_big, idx_small


def render_single_gaussian(
    xyz: torch.Tensor,
    cholesky: torch.Tensor,
    features_dc: torch.Tensor,
    opacity_raw: torch.Tensor,
    H: int,
    W: int,
    T: int,
    device: torch.device,
    background_zero: bool = True,
):
    """Render one Gaussian over full T. Returns (video [T,H,W,3], radius, num_tiles_hit)."""
    from gsplat.project_gaussians_video import project_gaussians_video
    from gsplat.rasterize_sum_video import rasterize_gaussians_sum_video

    BLOCK_W, BLOCK_H, BLOCK_T = 16, 16, 1
    tile_bounds = (
        (W + BLOCK_W - 1) // BLOCK_W,
        (H + BLOCK_H - 1) // BLOCK_H,
        (T + BLOCK_T - 1) // BLOCK_T,
    )
    means = xyz.float().contiguous()
    L_flat = cholesky.float().contiguous()
    xys, depths, radii, conics, num_tiles_hit = project_gaussians_video(
        means, L_flat, H, W, T, tile_bounds
    )
    # Colors in [0,1]; boost opacity so single-Gaussian splat is clearly visible
    colors = torch.sigmoid(features_dc).float().contiguous()
    opacity = torch.sigmoid(opacity_raw).clamp(min=0.95).contiguous()
    if background_zero:
        background = torch.zeros(3, dtype=torch.float32, device=device)
    else:
        background = torch.ones(3, dtype=torch.float32, device=device)
    out = rasterize_gaussians_sum_video(
        xys, depths, radii, conics, num_tiles_hit,
        colors, opacity, H, W, T,
        BLOCK_H, BLOCK_W, BLOCK_T,
        background=background, return_alpha=False,
    )
    out = torch.clamp(out, 0, 1)
    radius = radii[0].item() if radii.numel() else 0
    num_tiles = num_tiles_hit[0].item() if num_tiles_hit.numel() else 0
    return out, radius, num_tiles


def main():
    parser = argparse.ArgumentParser(description="Visualize Czz vs xy spread (single Gaussians, full T)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to GaussianVideo checkpoint")
    parser.add_argument("--out_dir", type=str, default="./viz_czz", help="Output directory")
    parser.add_argument("--H", type=int, default=1080)
    parser.add_argument("--W", type=int, default=1920)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--top_pct", type=float, default=15.0, help="Big = in top this %% by Cxx or Cyy (default 15)")
    parser.add_argument("--bottom_pct", type=float, default=15.0, help="Small = in bottom this %% by Cxx+Cyy (default 15)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint...")
    cholesky_elements, xyz_raw, features_dc, opacity_raw = load_checkpoint(args.checkpoint, device)
    cov_flat = cholesky_to_covariance(cholesky_elements)
    Cxx, Cyy, Czz = cov_flat[:, 0], cov_flat[:, 3], cov_flat[:, 5]
    N = cholesky_elements.shape[0]

    idx_big, idx_small = pick_big_small_xy(cov_flat, top_pct=args.top_pct, bottom_pct=args.bottom_pct)
    print(f"Big (top {args.top_pct}% by Cxx or Cyy) Gaussian index: {idx_big}  (Cxx={Cxx[idx_big].item():.6f}, Cyy={Cyy[idx_big].item():.6f}, Czz={Czz[idx_big].item():.6f})")
    print(f"Small (bottom {args.bottom_pct}% by Cxx+Cyy) Gaussian index: {idx_small}  (Cxx={Cxx[idx_small].item():.6f}, Cyy={Cyy[idx_small].item():.6f}, Czz={Czz[idx_small].item():.6f})")

    # Means in [-1,1]
    xyz = torch.tanh(xyz_raw)

    for label, idx in [("big_xy", idx_big), ("small_xy", idx_small)]:
        xyz_1 = xyz[idx : idx + 1].to(device)
        cholesky_1 = cholesky_elements[idx : idx + 1].to(device)
        features_1 = features_dc[idx : idx + 1].to(device)
        opacity_1 = opacity_raw[idx : idx + 1].to(device)

        for variant_factor, variant_label in zip(CZZ_VARIANTS, CZZ_VARIANT_LABELS):
            cholesky_scaled = scale_cholesky_czz(cholesky_1, variant_factor)
            video, radius, num_tiles = render_single_gaussian(
                xyz_1, cholesky_scaled, features_1, opacity_1,
                args.H, args.W, args.T, device, background_zero=True,
            )
            # video: [T, H, W, 3]
            subdir = out_dir / f"{label}_{variant_label}"
            subdir.mkdir(parents=True, exist_ok=True)
            for t in range(video.shape[0]):
                frame = (video[t].detach().cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(frame).save(subdir / f"frame_{t:04d}.png")
            print(f"  {label} {variant_label}: radius={radius}, num_tiles_hit={num_tiles}, saved to {subdir}")
            # Save a one-line summary for this variant
            with open(subdir / "info.txt", "w") as f:
                f.write(f"gaussian_index={idx} label={label} czz_factor={variant_factor}\nradius={radius}\nnum_tiles_hit={num_tiles}\n")

    print(f"\nDone. Outputs under {out_dir}")
    print("To change Czz factors, edit CZZ_VARIANTS and CZZ_VARIANT_LABELS at the top of this script.")


if __name__ == "__main__":
    main()
