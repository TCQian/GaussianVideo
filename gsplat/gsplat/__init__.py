from typing import Any
import torch
from .project_gaussians import project_gaussians
from .project_gaussians_video import project_gaussians_video
from .rasterize import rasterize_gaussians
from .project_gaussians_2d import project_gaussians_2d
from .project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from .rasterize_sum import rasterize_gaussians_sum
from .rasterize_sum_video import rasterize_gaussians_sum_video
from .utils import (
    map_gaussian_to_intersects,
    bin_and_sort_gaussians,
    compute_cumulative_intersects,
    compute_cov2d_bounds,
    get_tile_bin_edges,
    map_gaussian_to_intersects_video,
    get_tile_bin_edges_video,
)
from .sh import spherical_harmonics
from .version import __version__
import warnings


__all__ = [
    "__version__",
    "project_gaussians",
    "project_gaussians_video",
    "project_gaussians_2d",
    "project_gaussians_2d_scale_rot",
    "rasterize_gaussians",
    "rasterize_gaussians_sum",
    "rasterize_gaussians_sum_video",
    "spherical_harmonics",
    # utils
    "bin_and_sort_gaussians",
    "compute_cumulative_intersects",
    "compute_cov2d_bounds",
    "get_tile_bin_edges",
    "get_tile_bin_edges_video",
    "map_gaussian_to_intersects",
    "map_gaussian_to_intersects_video",
    # Function.apply() will be deprecated
    "ProjectGaussians",
    "ProjectGaussiansVideo",
    "ProjectGaussians2d",
    "ProjectGaussians2dScaleRot",
    "RasterizeGaussians",
    "RasterizeGaussiansSum",
    "RasterizeGaussiansSumVideo",
    "BinAndSortGaussians",
    "ComputeCumulativeIntersects",
    "ComputeCov2dBounds",
    "GetTileBinEdges",
    "MapGaussiansToIntersects",
    "MapGaussiansToIntersectsVideo",
    "SphericalHarmonics",
    "NDRasterizeGaussians",
]

# Define these for backwards compatibility


class MapGaussiansToIntersects(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "MapGaussiansToIntersects is deprecated, use map_gaussian_to_intersects instead",
            DeprecationWarning,
        )
        return map_gaussian_to_intersects(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError
    
class MapGaussiansToIntersectsVideo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "MapGaussiansToIntersectsVideo is deprecated, use map_gaussian_to_intersects_video instead",
            DeprecationWarning,
        )
        return map_gaussian_to_intersects_video(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class ComputeCumulativeIntersects(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "ComputeCumulativeIntersects is deprecated, use compute_cumulative_intersects instead",
            DeprecationWarning,
        )
        return compute_cumulative_intersects(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class ComputeCov2dBounds(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "ComputeCov2dBounds is deprecated, use compute_cov2d_bounds instead",
            DeprecationWarning,
        )
        return compute_cov2d_bounds(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class GetTileBinEdges(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "GetTileBinEdges is deprecated, use get_tile_bin_edges instead",
            DeprecationWarning,
        )
        return get_tile_bin_edges(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError
    
class GetTileBinEdgesVideo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "GetTileBinEdgesVideo is deprecated, use get_tile_bin_edges_video instead",
            DeprecationWarning,
        )
        return get_tile_bin_edges_video(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError

class BinAndSortGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "BinAndSortGaussians is deprecated, use bin_and_sort_gaussians instead",
            DeprecationWarning,
        )
        return bin_and_sort_gaussians(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class ProjectGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "ProjectGaussians is deprecated, use project_gaussians instead",
            DeprecationWarning,
        )
        return project_gaussians(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError
    
class ProjectGaussiansVideo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "ProjectGaussiansVideo is deprecated, use project_gaussians_video instead",
            DeprecationWarning,
        )
        return project_gaussians_video(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError

class ProjectGaussians2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "ProjectGaussians2d is deprecated, use project_gaussians_2d instead",
            DeprecationWarning,
        )
        return project_gaussians_2d(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError

class ProjectGaussians2dScaleRot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "ProjectGaussians2dScaleRot is deprecated, use project_gaussians_2d_scale_rot instead",
            DeprecationWarning,
        )
        return project_gaussians_2d_scale_rot(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


class RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "RasterizeGaussians is deprecated, use rasterize_gaussians instead",
            DeprecationWarning,
        )
        return rasterize_gaussians(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError

class RasterizeGaussiansSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "RasterizeGaussiansSum is deprecated, use rasterize_gaussians instead",
            DeprecationWarning,
        )
        return rasterize_gaussians_sum(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError
    
class RasterizeGaussiansSumVideo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "RasterizeGaussiansSumVideo is deprecated, use rasterize_gaussians_sum_video instead",
            DeprecationWarning,
        )
        return rasterize_gaussians_sum_video(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError

class NDRasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "NDRasterizeGaussians is deprecated, use rasterize_gaussians instead",
            DeprecationWarning,
        )
        return rasterize_gaussians(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError

class SphericalHarmonics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        warnings.warn(
            "SphericalHarmonics is deprecated, use spherical_harmonics instead",
            DeprecationWarning,
        )
        return spherical_harmonics(*args, **kwargs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError
