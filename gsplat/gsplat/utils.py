"""Python bindings for binning and sorting gaussians"""

from typing import Tuple

from jaxtyping import Float, Int
from torch import Tensor
import torch

import gsplat.cuda as _C


def map_gaussian_to_intersects(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
) -> Tuple[Float[Tensor, "cum_tiles_hit 1"], Float[Tensor, "cum_tiles_hit 1"]]:
    """Map each gaussian intersection to a unique tile ID and depth value for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): total number of tile intersections.
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor}:

        - **isect_ids** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit.
    """
    isect_ids, gaussian_ids = _C.map_gaussian_to_intersects(
        num_points,
        num_intersects,
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        cum_tiles_hit.contiguous(),
        tile_bounds,
    )
    return (isect_ids, gaussian_ids)

def map_gaussian_to_intersects_video(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
) -> Tuple[Float[Tensor, "cum_tiles_hit 1"], Float[Tensor, "cum_tiles_hit 1"]]:
    isect_ids, gaussian_ids = _C.map_gaussian_to_intersects_video(
        num_points,
        num_intersects,
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        cum_tiles_hit.contiguous(),
        tile_bounds,
    )
    return (isect_ids, gaussian_ids)


def get_tile_bin_edges(
    num_intersects: int, isect_ids_sorted: Int[Tensor, "num_intersects 1"]
) -> Int[Tensor, "num_intersects 2"]:
    """Map sorted intersection IDs to tile bins which give the range of unique gaussian IDs belonging to each tile.

    Expects that intersection IDs are sorted by increasing tile ID.

    Indexing into tile_bins[tile_idx] returns the range (lower,upper) of gaussian IDs that hit tile_idx.

    Note:
        This function is not differentiable to any input.

    Args:
        num_intersects (int): total number of gaussian intersects.
        isect_ids_sorted (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).

    Returns:
        A Tensor:

        - **tile_bins** (Tensor): range of gaussians IDs hit per tile.
    """
    return _C.get_tile_bin_edges(num_intersects, isect_ids_sorted.contiguous())


def get_tile_bin_edges_video(
    num_intersects: int, isect_ids_sorted: Int[Tensor, "num_intersects 1"], tile_bounds: Tuple[int, int, int],
) -> Int[Tensor, "num_intersects 2"]:
    return _C.get_tile_bin_edges_video(num_intersects, isect_ids_sorted.contiguous(), tile_bounds)

def compute_cov2d_bounds(
    cov2d: Float[Tensor, "batch 3"]
) -> Tuple[Float[Tensor, "batch_conics 3"], Float[Tensor, "batch_radii 1"]]:
    """Computes bounds of 2D covariance matrix

    Args:
        cov2d (Tensor): input cov2d of size  (batch, 3) of upper triangular 2D covariance values

    Returns:
        A tuple of {Tensor, Tensor}:

        - **conic** (Tensor): conic parameters for 2D gaussian.
        - **radii** (Tensor): radii of 2D gaussian projections.
    """
    assert (
        cov2d.shape[-1] == 3
    ), f"Expected input cov2d to be of shape (*batch, 3) (upper triangular values), but got {tuple(cov2d.shape)}"
    num_pts = cov2d.shape[0]
    assert num_pts > 0
    return _C.compute_cov2d_bounds(num_pts, cov2d.contiguous())


def compute_cumulative_intersects(
    num_tiles_hit: Float[Tensor, "batch 1"]
) -> Tuple[int, Float[Tensor, "batch 1"]]:
    """Computes cumulative intersections of gaussians. This is useful for creating unique gaussian IDs and for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_tiles_hit (Tensor): number of intersected tiles per gaussian.

    Returns:
        A tuple of {int, Tensor}:

        - **num_intersects** (int): total number of tile intersections.
        - **cum_tiles_hit** (Tensor): a tensor of cumulated intersections (used for sorting).
    """
    cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
    num_intersects = cum_tiles_hit[-1].item()
    return num_intersects, cum_tiles_hit


def bin_and_sort_gaussians(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
) -> Tuple[
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 2"],
]:
    """Mapping gaussians to sorted unique intersection IDs and tile bins used for fast rasterization.

    We return both sorted and unsorted versions of intersect IDs and gaussian IDs for testing purposes.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): cumulative number of total gaussian intersections
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **isect_ids_unsorted** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_unsorted** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **isect_ids_sorted** (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_sorted** (Tensor): sorted Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **tile_bins** (Tensor): range of gaussians hit per tile.
    """
    isect_ids, gaussian_ids = map_gaussian_to_intersects(
        num_points, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds
    )
    isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
    tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted)
    return isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins


# This function processes, sorts, and bins Gaussians based on the tiles they intersect with in a 3D spatial grid. The purpose is to prepare Gaussian splats for rendering or further computation by:

# Finding which tiles each Gaussian intersects with.
# Sorting these tile-Gaussian associations so that all Gaussians within the same tile are grouped together.
# Creating bin edges that allow efficient retrieval of Gaussians per tile.
# This is important for efficiency, as rendering or processing Gaussian splats requires knowing which Gaussians influence each tile.
def bin_and_sort_gaussians_video(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
) -> Tuple[
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 2"],
]:
    # Step 1: Compute tile intersections
    # isect_ids stores all the tile IDs that intersect with each Gaussian.
    # gaussian_ids stores the corresponding Gaussian index for each tile intersection.
    # isect_ids is of length h * w * t = 120 * 68 * 1 = 8160
    isect_ids, gaussian_ids = map_gaussian_to_intersects_video(
        num_points, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds
    )
    
    # Step 2: Sort tile intersections based on tile ID
    # The intersections (isect_ids) are sorted in ascending order.
    # sorted_indices contains the permutation indices that sort isect_ids.
    isect_ids_sorted, sorted_indices = torch.sort(isect_ids)

    # Step 3: Reorder the Gaussian IDs based on the sorted tile intersections.
    # The sorted_indices from the previous step are used to reorder gaussian_ids.
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
    
    # Step 4: Compute tile bin edges
    # tile_bins stores the start and end indices of Gaussian intersections for each tile.
    # For example, Tile 0: [0, 1], Tile 1: [1, 4]
    # This means Tile 0 has is intersected by Gaussian 0 to 1
    # And Tile 1 is intersected by Gaussian 1 to 4
    tile_bins = get_tile_bin_edges_video(num_intersects, isect_ids_sorted, tile_bounds)
    
    # Debugging prints to check sanity of outputs
    # print(f"[DEBUG] Number of intersections: {num_intersects}")
    # print(f"[DEBUG] isect_ids (first 10): {isect_ids[:10].tolist()}")
    # print(f"[DEBUG] gaussian_ids (first 10): {gaussian_ids[:10].tolist()}")
    # print(f"[DEBUG] isect_ids_sorted (first 10): {isect_ids_sorted[:10].tolist()}")
    # print(f"[DEBUG] gaussian_ids_sorted (first 10): {gaussian_ids_sorted[:10].tolist()}")
    # print(f"[DEBUG] tile_bins shape: {tile_bins.shape}")
    # print(f"[DEBUG] tile_bins (first 10): {tile_bins[:10].tolist()}")
    
    # Step 5: Return sorted and processed results
    
    # isect_ids: Tile IDs of all Gaussian intersections.
    # gaussian_ids: Original Gaussian IDs per tile.
    # isect_ids_sorted: Sorted tile IDs for efficient access.
    # gaussian_ids_sorted: Sorted Gaussian IDs matching isect_ids_sorted.
    # tile_bins: Start/end indices for each tile in isect_ids_sorted.
    return isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins
