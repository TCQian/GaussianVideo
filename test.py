"""
Analyze GaussianVideo checkpoint: distributions of Cholesky elements,
covariance elements (after LL^T), conic (Sigma^{-1}), scales, and radii.
Picks one Gaussian per scale bin (small/medium/large) and reports covariance,
conic, and the range of delta that decays power to 0.

Usage:
  python test.py --checkpoint path/to/gaussian_model.pth.tar [--H 1080 --W 1920 --T 50] [--no-radii]
  python test.py --checkpoint path/to/layer_0_model.pth.tar  # 3D2D format also supported
"""
import os
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


# Cholesky bound used in GaussianVideo (same as in gaussianvideo.py)
CHOLESKY_BOUND = torch.tensor([0.5, 0, 0, 0.5, 0, 0.5], dtype=torch.float32)


def load_checkpoint(path: str, device="cpu"):
    """Load checkpoint; support both GaussianVideo (_xyz, _cholesky) and 3D2D (_xyz_3D, _cholesky_3D) formats."""
    ckpt = torch.load(path, map_location=device)
    if "_xyz_3D" in ckpt:
        cholesky_raw = ckpt["_cholesky_3D"]
        xyz_raw = ckpt["_xyz_3D"]
        fmt = "3D2D"
    elif "_xyz" in ckpt:
        cholesky_raw = ckpt["_cholesky"]
        xyz_raw = ckpt["_xyz"]
        fmt = "GaussianVideo"
    else:
        raise KeyError("Checkpoint must contain _xyz / _cholesky (GaussianVideo) or _xyz_3D / _cholesky_3D (3D2D)")
    bound = CHOLESKY_BOUND.to(cholesky_raw.device).view(1, 6)
    cholesky_elements = cholesky_raw + bound  # (N, 6)
    return cholesky_elements, xyz_raw, fmt


def cholesky_to_covariance(L_flat: torch.Tensor):
    """
    L_flat: (N, 6) = [l11, l21, l31, l22, l32, l33] (lower triangular row-major).
    Returns cov_flat (N, 6) = [Cxx, Cxy, Cxz, Cyy, Cyz, Czz] (upper triangular).
    Sigma = L @ L^T.
    """
    N = L_flat.shape[0]
    device = L_flat.device
    l11, l21, l31 = L_flat[:, 0], L_flat[:, 1], L_flat[:, 2]
    l22, l32, l33 = L_flat[:, 3], L_flat[:, 4], L_flat[:, 5]

    Cxx = l11 * l11
    Cxy = l11 * l21
    Cxz = l11 * l31
    Cyy = l21 * l21 + l22 * l22
    Cyz = l21 * l31 + l22 * l32
    Czz = l31 * l31 + l32 * l32 + l33 * l33

    cov_flat = torch.stack([Cxx, Cxy, Cxz, Cyy, Cyz, Czz], dim=1)
    return cov_flat


def covariance_diagonal_scales(cov_flat: torch.Tensor):
    """Return sqrt of diagonal of Sigma: (scale_x, scale_y, scale_z) as (N, 3)."""
    Cxx, Cyy, Czz = cov_flat[:, 0], cov_flat[:, 3], cov_flat[:, 5]
    scale_x = torch.sqrt(Cxx.clamp(min=1e-10))
    scale_y = torch.sqrt(Cyy.clamp(min=1e-10))
    scale_z = torch.sqrt(Czz.clamp(min=1e-10))
    return torch.stack([scale_x, scale_y, scale_z], dim=1)


def cov_flat_to_matrix(cov_flat: torch.Tensor, idx: int) -> torch.Tensor:
    """Single Gaussian: cov_flat (N,6) -> 3x3 symmetric Sigma at index idx."""
    c = cov_flat[idx]
    return torch.tensor(
        [
            [c[0].item(), c[1].item(), c[2].item()],
            [c[1].item(), c[3].item(), c[4].item()],
            [c[2].item(), c[4].item(), c[5].item()],
        ],
        dtype=cov_flat.dtype,
        device=cov_flat.device,
    )


def covariance_to_conic_flat(cov_flat: torch.Tensor) -> torch.Tensor:
    """
    For each Gaussian: Sigma (from cov_flat) -> conic = Sigma^{-1}.
    Returns (N, 6) upper triangular [invCxx, invCxy, invCxz, invCyy, invCyz, invCzz].
    Batched for speed.
    """
    N = cov_flat.shape[0]
    S = torch.zeros(N, 3, 3, dtype=cov_flat.dtype, device=cov_flat.device)
    S[:, 0, 0] = cov_flat[:, 0]
    S[:, 0, 1] = S[:, 1, 0] = cov_flat[:, 1]
    S[:, 0, 2] = S[:, 2, 0] = cov_flat[:, 2]
    S[:, 1, 1] = cov_flat[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = cov_flat[:, 4]
    S[:, 2, 2] = cov_flat[:, 5]
    try:
        conic_3x3 = torch.linalg.inv(S)
    except Exception:
        conic_3x3 = torch.full_like(S, float("nan"))
    conic_flat = torch.stack(
        [
            conic_3x3[:, 0, 0], conic_3x3[:, 0, 1], conic_3x3[:, 0, 2],
            conic_3x3[:, 1, 1], conic_3x3[:, 1, 2], conic_3x3[:, 2, 2],
        ],
        dim=1,
    )
    return conic_flat


def pick_example_gaussian_indices(scale_xyz: np.ndarray) -> tuple:
    """
    Bin by overall scale (geometric mean of scale_x, scale_y, scale_z).
    Return one index from each bin: small (0-33p), medium (33-66p), large (66-100p).
    """
    # overall scale per Gaussian (geometric mean of the 3 scales)
    scale_geom = np.exp(np.mean(np.log(scale_xyz + 1e-10), axis=1))
    p33 = np.percentile(scale_geom, 33)
    p66 = np.percentile(scale_geom, 66)
    small_mask = scale_geom <= p33
    medium_mask = (scale_geom > p33) & (scale_geom <= p66)
    large_mask = scale_geom > p66
    # pick median index in each bin (by scale_geom)
    def pick_one(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return None
        s = scale_geom[idx]
        return idx[np.argmin(np.abs(s - np.median(s)))]
    return pick_one(small_mask), pick_one(medium_mask), pick_one(large_mask)


def format_gaussian_example(
    idx: int,
    cov_flat: np.ndarray,
    conic_flat: np.ndarray,
    scale_xyz: np.ndarray,
    power_threshold: float = 1e-6,
) -> str:
    """Format one Gaussian: 3x3 covariance, 3x3 conic, and range of delta for power decay to threshold."""
    lines = []
    c = cov_flat[idx]
    Sigma = np.array([[c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], c[5]]])
    invc = conic_flat[idx]
    Conic = np.array([[invc[0], invc[1], invc[2]], [invc[1], invc[3], invc[4]], [invc[2], invc[4], invc[5]]])
    lines.append(f"  Covariance Sigma (3x3):")
    for row in Sigma:
        lines.append(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")
    lines.append(f"  Conic Sigma^{{-1}} (3x3):")
    for row in Conic:
        lines.append(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")
    lines.append(f"  Covariance scales (sqrt diag): [{scale_xyz[idx, 0]:.6f}, {scale_xyz[idx, 1]:.6f}, {scale_xyz[idx, 2]:.6f}]")
    decay = conic_decay_delta(Conic, power_threshold=power_threshold)
    lines.append(f"  Power decay to {power_threshold:.0e}:")
    deltas_str = ", ".join(f"{x:.6f}" for x in decay["principal_deltas"])
    lines.append(f"    Principal-axis deltas (distance at which weight = {power_threshold:.0e}): [{deltas_str}]")
    lines.append(f"    Max delta (effective radius): {decay['max_delta']:.6f}")
    lines.append(f"    Min delta: {decay['min_delta']:.6f}")
    return "\n".join(lines)


def conic_decay_delta(conic_3x3: np.ndarray, power_threshold: float = 1e-6) -> dict:
    """
    Gaussian weight = exp(-0.5 * d^T @ conic @ d). Find range of delta (distance from center)
    such that power decays to power_threshold.
    conic = Sigma^{-1}. Returns principal-axis deltas and max delta.
    """
    # Eigenvalues of conic = 1/variance along each principal axis.
    eigvals, eigvecs = np.linalg.eigh(conic_3x3)
    eigvals = np.maximum(eigvals, 1e-12)
    # d^T conic d = 2*ln(1/threshold) => along axis i: delta_i^2 * eigval_i = 2*ln(1/t) => delta_i = sqrt(2*ln(1/t)/eigval_i)
    k = 2.0 * np.log(1.0 / power_threshold)
    principal_deltas = np.sqrt(k / eigvals)
    return {
        "principal_deltas": principal_deltas,
        "max_delta": float(np.max(principal_deltas)),
        "min_delta": float(np.min(principal_deltas)),
        "eigenvalues_conic": eigvals,
    }


def compute_projected_radii(xyz: torch.Tensor, cholesky_elements: torch.Tensor, H: int, W: int, T: int):
    """Run project_gaussians_video to get radii (in tile/pixel space)."""
    from gsplat.project_gaussians_video import project_gaussians_video

    BLOCK_W, BLOCK_H, BLOCK_T = 16, 16, 1
    tile_bounds = (
        (W + BLOCK_W - 1) // BLOCK_W,
        (H + BLOCK_H - 1) // BLOCK_H,
        (T + BLOCK_T - 1) // BLOCK_T,
    )
    # xyz in model is tanh(_xyz) in [-1,1]; project expects same
    means = xyz.float().contiguous()
    L_flat = cholesky_elements.float().contiguous()
    xys, depths, radii, conics, num_tiles_hit = project_gaussians_video(
        means, L_flat, H, W, T, tile_bounds
    )
    return radii  # (N,) int


def plot_histograms(
    cholesky_elements: np.ndarray,
    cov_elements: np.ndarray,
    scale_xyz: np.ndarray,
    radii: Optional[np.ndarray],
    out_dir: Path,
    prefix: str = "",
    conic_elements: Optional[np.ndarray] = None,
):
    """Draw and save histograms for Cholesky, covariance, conic, scales, and radii."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chol_names = ["l11", "l21", "l31", "l22", "l32", "l33"]
    cov_names = ["Cxx", "Cxy", "Cxz", "Cyy", "Cyz", "Czz"]
    conic_names = ["invCxx", "invCxy", "invCxz", "invCyy", "invCyz", "invCzz"]
    scale_names = ["scale_x", "scale_y", "scale_z"]

    def _hist(data, title, xlabel, fname, bins=80):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.hist(data.flatten(), bins=bins, density=True, alpha=0.8, color="steelblue", edgecolor="black", linewidth=0.3)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

    # Cholesky elements
    for i, name in enumerate(chol_names):
        _hist(
            cholesky_elements[:, i],
            f"Cholesky element: {name}",
            name,
            f"{prefix}hist_cholesky_{name}.png",
        )

    # Covariance elements
    for i, name in enumerate(cov_names):
        _hist(
            cov_elements[:, i],
            f"Covariance element (after LL^T): {name}",
            name,
            f"{prefix}hist_cov_{name}.png",
        )

    # Conic elements (Sigma^{-1})
    if conic_elements is not None:
        valid = np.isfinite(conic_elements).all(axis=1)
        if np.any(valid):
            conic_valid = conic_elements[valid]
            for i, name in enumerate(conic_names):
                _hist(
                    conic_valid[:, i],
                    f"Conic element (Sigma^{{-1}}): {name}",
                    name,
                    f"{prefix}hist_conic_{name}.png",
                )

    # Covariance scale (sqrt of diagonal)
    for i, name in enumerate(scale_names):
        _hist(
            scale_xyz[:, i],
            f"Covariance scale (sqrt diag): {name}",
            name,
            f"{prefix}hist_scale_{name}.png",
        )

    # Combined scale histogram (all 3 axes)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i, name in enumerate(scale_names):
        ax.hist(scale_xyz[:, i], bins=80, density=True, alpha=0.5, label=name, histtype="step", linewidth=1.5)
    ax.set_title("Covariance scales (sqrt of diagonal)")
    ax.set_xlabel("Scale")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}hist_scale_all.png", dpi=150)
    plt.close()

    if radii is not None:
        radii_flat = radii.flatten()
        radii_flat = radii_flat[radii_flat > 0]  # exclude zero (invisible)
        if len(radii_flat) > 0:
            _hist(
                radii_flat,
                "Projected radii (from project_gaussians_video)",
                "Radius (tile/pixel)",
                f"{prefix}hist_radii.png",
                bins=min(80, int(radii_flat.max()) + 1),
            )

    # Combined overview: Cholesky diagonal (l11,l22,l33), cov scales, and radii
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    chol_diag = np.stack([cholesky_elements[:, 0], cholesky_elements[:, 3], cholesky_elements[:, 5]], axis=1)
    for i, (ax, name) in enumerate(zip(axes, ["Cholesky diag (l11,l22,l33)", "Cov scales (sqrt diag)", "Radii"])):
        if name == "Cholesky diag (l11,l22,l33)":
            for j, lab in enumerate(["l11", "l22", "l33"]):
                ax.hist(chol_diag[:, j], bins=60, density=True, alpha=0.5, label=lab, histtype="step", linewidth=1.2)
        elif name == "Cov scales (sqrt diag)":
            for j, lab in enumerate(scale_names):
                ax.hist(scale_xyz[:, j], bins=60, density=True, alpha=0.5, label=lab, histtype="step", linewidth=1.2)
        else:
            if radii is not None and radii.size > 0:
                r = radii.flatten()
                r = r[r > 0]
            else:
                r = np.array([])
            if len(r) > 0:
                ax.hist(r, bins=min(60, int(r.max()) + 1), density=True, alpha=0.8, color="green", edgecolor="black")
            else:
                ax.text(0.5, 0.5, "N/A (no radii)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(name)
        ax.set_xlabel(name.split(" ")[0] if " " in name else name)
        ax.set_ylabel("Density")
        if name != "Radii":
            ax.legend()
    plt.suptitle("Gaussian shape overview", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}overview.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze GaussianVideo checkpoint distributions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth.tar)")
    parser.add_argument("--out_dir", type=str, default="./analysis_out", help="Output directory for histograms")
    parser.add_argument("--H", type=int, default=1080, help="Video height (for projected radii)")
    parser.add_argument("--W", type=int, default=1920, help="Video width (for projected radii)")
    parser.add_argument("--T", type=int, default=50, help="Video length (for projected radii)")
    parser.add_argument("--no-radii", action="store_true", help="Skip projected radii (no gsplat projection)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:0 or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)

    print(f"Loading checkpoint: {args.checkpoint}")
    cholesky_elements, xyz_raw, fmt = load_checkpoint(args.checkpoint, device)
    N = cholesky_elements.shape[0]
    print(f"Format: {fmt}, num_gaussians: {N}")

    # Cholesky elements (already with bound applied in load_checkpoint)
    cholesky_np = cholesky_elements.detach().cpu().numpy()

    # Covariance from L @ L^T
    cov_flat = cholesky_to_covariance(cholesky_elements)
    cov_np = cov_flat.detach().cpu().numpy()

    # Covariance scales = sqrt(diagonal)
    scale_xyz = covariance_diagonal_scales(cov_flat)
    scale_np = scale_xyz.detach().cpu().numpy()

    # Conic = Sigma^{-1} (for decay visualization and example report)
    conic_flat = covariance_to_conic_flat(cov_flat)
    conic_np = conic_flat.detach().cpu().numpy()

    # Projected radii (optional)
    radii_np = None
    if not args.no_radii:
        try:
            xyz = torch.tanh(xyz_raw).to(device)
            radii = compute_projected_radii(
                xyz, cholesky_elements.to(device), args.H, args.W, args.T
            )
            radii_np = radii.detach().cpu().numpy()
            print(f"Radii: min={radii_np.min()}, max={radii_np.max()}, mean={radii_np[radii_np > 0].mean():.2f}")
        except Exception as e:
            print(f"Could not compute projected radii (need gsplat): {e}")
            radii_np = None

    # Summary stats (build lines for file and print)
    lines = [f"Checkpoint: {args.checkpoint}", f"Format: {fmt}, N = {N}", ""]
    lines.append("Cholesky elements (with bound):")
    for i, name in enumerate(["l11", "l21", "l31", "l22", "l32", "l33"]):
        col = cholesky_np[:, i]
        s = f"  {name}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}"
        lines.append(s)
    lines.append("")
    lines.append("Covariance elements (LL^T):")
    for i, name in enumerate(["Cxx", "Cxy", "Cxz", "Cyy", "Cyz", "Czz"]):
        col = cov_np[:, i]
        s = f"  {name}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}"
        lines.append(s)
    lines.append("")
    lines.append("Covariance scales (sqrt diag):")
    for i, name in enumerate(["scale_x", "scale_y", "scale_z"]):
        col = scale_np[:, i]
        s = f"  {name}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}"
        lines.append(s)
    if radii_np is not None:
        r = radii_np[radii_np > 0]
        lines.append("")
        lines.append(f"Projected radii: min={radii_np.min()}, max={radii_np.max()}, mean(>0)={r.mean():.2f}")

    # Conic (Sigma^{-1}) stats
    conic_valid = conic_np[np.isfinite(conic_np).all(axis=1)]
    if len(conic_valid) > 0:
        lines.append("")
        lines.append("Conic elements (Sigma^{-1}):")
        for i, name in enumerate(["invCxx", "invCxy", "invCxz", "invCyy", "invCyz", "invCzz"]):
            col = conic_valid[:, i]
            s = f"  {name}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}"
            lines.append(s)

    for s in lines:
        print(s)

    # eg. /home/e/e0407638/github/GaussianVideo/checkpoints/${DATA_NAME}/GaussianVideo_i${TRAIN_ITERATIONS}_g${NUM_POINTS}_f${NUM_FRAMES}_s${START_FRAME}/${DATA_NAME}
    # get GaussianVideo_i${TRAIN_ITERATIONS}_g${NUM_POINTS}_f${NUM_FRAMES}_s${START_FRAME}_${DATA_NAME}
    list = os.path.dirname(args.checkpoint).split("/")
    file_name = list[-2] + "_" + list[-1]
    summary_path = out_dir / f"{file_name}_summary.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSummary written to: {summary_path}")

    # Example Gaussians: one from small / medium / large scale bin (covariance, conic, delta range)
    power_threshold = 1e-6
    idx_small, idx_medium, idx_large = pick_example_gaussian_indices(scale_np)
    example_lines = [
        "Example Gaussians (one per scale bin: small / medium / large)",
        "Power decay: weight = exp(-0.5 * d^T conic d); delta = distance at which weight reaches threshold.",
        f"Threshold = {power_threshold:.0e}",
        "",
    ]
    for label, idx in [("Small scale (0-33 pct)", idx_small), ("Medium scale (33-66 pct)", idx_medium), ("Large scale (66-100 pct)", idx_large)]:
        if idx is None:
            example_lines.append(f"{label}: no Gaussian in bin")
            example_lines.append("")
            continue
        example_lines.append(f"--- {label} (index {idx}) ---")
        example_lines.append(format_gaussian_example(idx, cov_np, conic_np, scale_np, power_threshold=power_threshold))
        example_lines.append("")
    example_text = "\n".join(example_lines)
    print(example_text)
    examples_path = out_dir / f"{Path(args.checkpoint).stem}_example_gaussians.txt"
    with open(examples_path, "w") as f:
        f.write(example_text)
    print(f"Example Gaussians report written to: {examples_path}")

    # Histograms (including conic)
    prefix = Path(args.checkpoint).stem + "_"
    plot_histograms(cholesky_np, cov_np, scale_np, radii_np, out_dir, prefix=prefix, conic_elements=conic_np)
    print(f"Histograms saved to: {out_dir}")


if __name__ == "__main__":
    main()
