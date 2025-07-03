"""Python bindings for 3D gaussian projection"""

from typing import Tuple

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C


def project_gaussians_video(
    means3d: Float[Tensor, "*batch 2"],
    L_elements: Float[Tensor, "*batch 3"],
    img_height: int,
    img_width: int,
    video_length: int,
    tile_bounds: Tuple[int, int, int],
    timestamp: int = 0,
    clip_thresh: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
    """This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Note:
        This function is differentiable w.r.t the means3d, scales and quats inputs.

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (float): A global scaling factor applied to the scene.
       quats (Tensor): rotations in quaternion [w,x,y,z] format.
       viewmat (Tensor): view matrix for rendering.
       projmat (Tensor): projection matrix for rendering.
       fx (float): focal length x.
       fy (float): focal length y.
       cx (float): principal point x.
       cy (float): principal point y.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).
       clip_thresh (float): minimum z depth threshold.

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **conics** (Tensor): conic parameters for 2D gaussian.
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
    """
    return _ProjectGaussiansVideo.apply(
        means3d.contiguous(),
        L_elements.contiguous(),
        img_height,
        img_width,
        video_length,
        tile_bounds,
        timestamp,
        clip_thresh,
    )

class _ProjectGaussiansVideo(Function):
    """Project 3D gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means3d: Float[Tensor, "*batch 2"],
        L_elements: Float[Tensor, "*batch 3"],
        img_height: int,
        img_width: int,
        video_length: int,
        tile_bounds: Tuple[int, int, int],
        timestamp: int = 0,
        clip_thresh: float = 0.01,
    ):
        # means2d.shape is (9000, 3)
        # num_points = 9000 (number of gaussians)
        num_points = means3d.shape[-2]

        (
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
        ) = _C.project_gaussians_video_forward(
            num_points,
            means3d,
            L_elements,
            img_height,
            img_width,
            video_length,
            tile_bounds,
            timestamp,
            clip_thresh,
        )
        
        # Save non-tensors.
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.video_length = video_length
        ctx.num_points = num_points
        ctx.timestamp = timestamp

        # Save tensors.
        ctx.save_for_backward(
            means3d,
            L_elements,
            radii,
            conics,
        )

        return (xys, depths, radii, conics, num_tiles_hit)

    @staticmethod
    def backward(ctx, v_xys, v_depths, v_radii, v_conics, v_num_tiles_hit):
        (
            means3d,
            L_elements,
            radii,
            conics,
        ) = ctx.saved_tensors

        (v_cov3d, v_mean3d, v_L_elements) = _C.project_gaussians_video_backward(
            ctx.num_points,
            means3d,
            L_elements,
            ctx.img_height,
            ctx.img_width,
            ctx.video_length,
            ctx.timestamp,
            radii,
            conics,
            v_xys,
            v_depths,
            v_conics,
        )

        # Return a gradient for each input.
        return (
            # means3d: Float[Tensor, "*batch 3"],
            v_mean3d,
            # scales: Float[Tensor, "*batch 3"],
            v_L_elements,
            # img_height: int,
            None,
            # img_width: int,
            None,
            # video_length: int,
            None,
            # tile_bounds: Tuple[int, int, int],
            None,
            # timestamp: int,
            None,
            # clip_thresh,
            None,
        )
